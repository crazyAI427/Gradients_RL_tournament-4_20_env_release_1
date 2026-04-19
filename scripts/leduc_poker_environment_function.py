import os
import re
import random
import requests
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from trl.experimental.openenv import generate_rollout_completions


# ---------------------------------------------------------------------------
# Leduc Poker Constants
# ---------------------------------------------------------------------------

# Card values for deadwood-equivalent (hand strength proxy)
# In Leduc: J=1, Q=2, K=3 (higher is better)
CARD_RANK_VALUE = {'J': 1, 'Q': 2, 'K': 3}

# Reward constants — aligns with validator "higher is better" scoring
TERMINAL_WIN_REWARD = 1.0
TERMINAL_LOSS_REWARD = -1.0
PAIR_BONUS = 0.15           # extra for winning with a pair (strongest hand)
HIGHCARD_WIN_BONUS = 0.05   # extra for winning with high card
INVALID_PENALTY = -0.1
INVALID_TOTAL_CLIP = -0.3
TERMINAL_REWARD_CLIP = 1.0  # final clip for validator alignment

# Chip improvement shaping weight
CHIP_WEIGHT = 0.3  # fraction of reward from chip improvement


# ---------------------------------------------------------------------------
# Card Utilities
# ---------------------------------------------------------------------------

def card_id_to_name(card_id: int) -> str:
    """Convert OpenSpiel card ID to human-readable name.
    2-player (6 cards): 0=J♠ 1=J♥ 2=Q♠ 3=Q♥ 4=K♠ 5=K♥
    """
    ranks = ['J', 'Q', 'K']
    suits = ['s', 'h']
    rank_idx = card_id // 2
    suit_idx = card_id % 2
    if rank_idx < len(ranks):
        return f"{ranks[rank_idx]}{suits[suit_idx]}"
    return f"Card{card_id}"


def card_rank(card_name: str) -> int:
    """Return numeric rank value of a card name like 'Qs' or 'Kh'."""
    if not card_name:
        return 0
    rank_char = card_name[0].upper()
    return CARD_RANK_VALUE.get(rank_char, 0)


def is_pair(private_card: str, public_card: str) -> bool:
    """True if private and public cards share the same rank."""
    if not private_card or not public_card:
        return False
    return private_card[0].upper() == public_card[0].upper()


# ---------------------------------------------------------------------------
# GameState for Leduc Poker
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Leduc Poker game state (mirrors gin rummy GameState interface)."""
    private_card: str        # e.g. 'Qs'
    public_card: str         # e.g. 'Kh' or '' if not revealed
    round_num: int           # 1 or 2
    pot: int                 # current pot size
    my_chips: int            # player's remaining chips
    opp_chips: int           # opponent's remaining chips
    phase: str               # 'Round1' | 'Round2' | 'Showdown'
    player_id: int           # 0 or 1
    round1_seq: str          # betting sequence round 1
    round2_seq: str          # betting sequence round 2
    # Kept for interface compatibility with gin rummy reward calculator
    deadwood: int = 0        # unused; always 0

    def has_pair(self) -> bool:
        return is_pair(self.private_card, self.public_card)

    def hand_strength(self) -> int:
        """Rough hand strength: pair > high card rank."""
        if self.has_pair():
            return 10 + card_rank(self.private_card)
        return card_rank(self.private_card)

    def total_chips(self) -> int:
        return self.my_chips


# ---------------------------------------------------------------------------
# Observation Parsing
# ---------------------------------------------------------------------------

def _extract_field(info_str: str, pattern: str) -> str:
    match = re.search(pattern, info_str)
    return match.group(1) if match else ""


def _parse_betting_sequence(sequence_str: str) -> str:
    actions_map = {0: "Fold", 1: "Call/Check", 2: "Raise"}
    if not sequence_str or sequence_str.strip() == "":
        return "(no actions yet)"
    numbers = [int(x) for x in sequence_str.split() if x.isdigit()]
    if not numbers:
        return "(no actions yet)"
    return ", ".join(actions_map.get(a, f"Action{a}") for a in numbers)


def extract_and_format_observation(obs_text: str) -> str:
    """
    Pass-through for Leduc Poker — the server already provides clean observations.
    Mirrors the gin rummy function signature; strips game-rules preamble if present.
    """
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text

    # Strip any preamble before 'Current State:' if present
    state_match = re.search(r'Current State:\n', obs_text)
    if state_match:
        state_text = obs_text[state_match.start():]
        player_match = re.search(r'You are Player (\d+)', obs_text)
        player_id = int(player_match.group(1)) if player_match else 0
        if 'Legal Actions:' in state_text:
            before, after = state_text.split('Legal Actions:', 1)
            return before + f"You are Player {player_id}.\nLegal Actions:" + after
        return state_text

    return obs_text


def parse_game_state(observation: str) -> GameState:
    """
    Parse a Leduc Poker observation string into a GameState.

    The server observation format includes fields like:
      Your card: Qs
      Public card: Kh
      Current round: 2/2
      Pot size: 8 chips
      Your chips: 95
      Opponent chips: 97
      Round 1 betting: Call/Check, Raise, Call/Check
      ...
      You are Player 0.
    """
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response — not a game state")

    def _get(pattern, default=""):
        m = re.search(pattern, observation)
        return m.group(1).strip() if m else default

    private_card = _get(r'Your card:\s*(\S+)')
    public_card   = _get(r'Public card:\s*(\S+)')
    round_num     = int(_get(r'Current round:\s*(\d+)', '1'))
    pot           = int(_get(r'Pot size:\s*(\d+)', '0'))
    my_chips      = int(_get(r'Your chips:\s*(\d+)', '100'))
    opp_chips     = int(_get(r'Opponent chips:\s*(\d+)', '100'))
    round1_seq    = _get(r'Round 1 betting:\s*(.*)')
    round2_seq    = _get(r'Round 2 betting:\s*(.*)')
    player_id     = int(_get(r'You are Player\s*(\d+)', '0'))

    phase = 'Round2' if round_num == 2 else 'Round1'

    return GameState(
        private_card=private_card,
        public_card=public_card,
        round_num=round_num,
        pot=pot,
        my_chips=my_chips,
        opp_chips=opp_chips,
        phase=phase,
        player_id=player_id,
        round1_seq=round1_seq,
        round2_seq=round2_seq,
    )


# ---------------------------------------------------------------------------
# Reward Calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Chip-based reward calculator for Leduc Poker, normalized to [-1, 1].

    Components:
    - Terminal: +1.0 win (+ optional pair/highcard bonus), -1.0 loss
    - Chip improvement: weighted fraction of chip gain vs starting stack
    - Invalid action penalty: -0.1 per step, total clipped to INVALID_TOTAL_CLIP
    """

    def __init__(self):
        self.invalid_penalty = INVALID_PENALTY

    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float,
        is_invalid: bool = False,
    ) -> float:
        """Per-step reward: only invalid action penalty (no dense shaping mid-game)."""
        if is_invalid:
            return self.invalid_penalty
        return 0.0

    def calculate_episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        initial_state: Optional[GameState],
        final_state: Optional[GameState],
        all_states: Optional[list[GameState]] = None,
    ) -> float:
        """
        Combine chip improvement + terminal bonus + invalid penalties.
        Clipped to [-1, 1] for validator alignment.
        """
        # 1. Chip improvement component
        chip_component = 0.0
        if initial_state and final_state:
            start_chips = initial_state.my_chips
            end_chips   = final_state.my_chips
            if start_chips > 0:
                chip_delta = (end_chips - start_chips) / start_chips
                chip_component = chip_delta * CHIP_WEIGHT

        # 2. Terminal bonus
        terminal = 0.0
        if done:
            if env_reward > 0.5:
                terminal = TERMINAL_WIN_REWARD
                # Bonus for winning with a pair (strongest hand type)
                if final_state and final_state.has_pair():
                    terminal += PAIR_BONUS
                else:
                    terminal += HIGHCARD_WIN_BONUS
            else:
                terminal = TERMINAL_LOSS_REWARD

        # 3. Invalid action penalty (accumulated, clipped)
        invalid_total = sum(r for r in step_rewards if r < 0)
        invalid_total = max(invalid_total, INVALID_TOTAL_CLIP)

        raw = chip_component + terminal + invalid_total
        return max(min(raw, TERMINAL_REWARD_CLIP), -TERMINAL_REWARD_CLIP)


# ---------------------------------------------------------------------------
# Action Extraction
# ---------------------------------------------------------------------------

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]


def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def extract_action_id(completion_text: str) -> str:
    """Extract a clean numeric action ID from model completion text."""
    cleaned = remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5].strip()
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    match = re.search(r"-?\d+", cleaned)
    return match.group(0) if match else cleaned.strip()


# ---------------------------------------------------------------------------
# Curriculum Scheduler (identical interface to gin rummy)
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Manages curriculum learning parameters throughout training."""

    def __init__(
        self,
        initial_max_turn=1,
        final_max_turn=8,       # Leduc max game length ~8 actions
        rollouts_per_stage=1280,
        initial_hint_prob=0.8,
        final_hint_prob=0.0,
        hint_decay_optimizer_steps=100,
        warmup_rollouts=128,
        mcts_warmup_optimizer_steps=None,
        initial_mcts_sims=5,
        final_mcts_sims=25,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.hint_decay_optimizer_steps = hint_decay_optimizer_steps
        self.warmup_rollouts = warmup_rollouts
        self.mcts_warmup_optimizer_steps = (
            0 if mcts_warmup_optimizer_steps is None else mcts_warmup_optimizer_steps
        )
        self.initial_mcts_sims = initial_mcts_sims
        self.final_mcts_sims = final_mcts_sims
        self.total_rollouts = 0

    def get_max_turn(self):
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted = self.total_rollouts - self.warmup_rollouts
        stage = adjusted // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self, optimizer_step: Optional[int] = None):
        current_step = 0 if optimizer_step is None else optimizer_step
        if self.hint_decay_optimizer_steps <= 0:
            return self.final_hint_prob
        progress = min(max(current_step, 0) / self.hint_decay_optimizer_steps, 1.0)
        current_prob = self.initial_hint_prob - progress * (
            self.initial_hint_prob - self.final_hint_prob
        )
        return max(current_prob, self.final_hint_prob)

    def get_mcts_sims(self, optimizer_step: Optional[int] = None):
        current_step = 0 if optimizer_step is None else optimizer_step
        if self.mcts_warmup_optimizer_steps <= 0:
            return self.final_mcts_sims
        progress = min(max(current_step, 0) / self.mcts_warmup_optimizer_steps, 1.0)
        return int(
            self.initial_mcts_sims
            + progress * (self.final_mcts_sims - self.initial_mcts_sims)
        )

    def step(self, num_rollouts=1):
        self.total_rollouts += num_rollouts

    def get_status(self, optimizer_step: Optional[int] = None):
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(optimizer_step),
            "mcts_sims": self.get_mcts_sims(optimizer_step),
        }


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

LEDUC_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits × (num_players + 1) ranks. For 2 players: 6 cards (J♠ J♥ Q♠ Q♥ K♠ K♥).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), "
    "Call/Check (match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n"
    "\n# Output Format",
    "You must respond with ONLY the action ID (a single number).",
    "Do NOT include descriptions or explanations.",
    "Examples:\n"
    '- For action "0 -> fold": respond "0"\n'
    '- For action "1 -> call": respond "1"\n'
    '- For action "2 -> raise": respond "2"'
)

LEDUC_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "- Round 1: Raise with K, call with Q, consider folding J against a raise\n"
    "- Round 2: If you have a pair, you have the strongest hand — raise aggressively\n"
    "- If public card matches your private card rank, you have a pair (unbeatable)\n"
    "- Fold when facing a raise with a weak hand (J, no pair) to save chips\n"
    "- Bluff sparingly — opponent has limited information too\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
)


# ---------------------------------------------------------------------------
# Shared Initialization Helper
# ---------------------------------------------------------------------------

GAMES_TO_TASK_ID_RANGE = {
    "goofspiel":    (0,         99999999),
    "liars_dice":   (100000000, 199999999),
    "leduc_poker":  (200000000, 299999999),
    "gin_rummy":    (300000000, 399999999),
    "othello":      (400000000, 499999999),
    "backgammon":   (500000000, 599999999),
    "hex":          (600000000, 699999999),
    "clobber":      (700000000, 799999999),
}

SELECTED_GAME = "leduc_poker"
TIMEOUT = 2400


def _init_static(fn, trainer):
    """One-time initialization stored as function attributes (one per rank)."""
    if getattr(fn, "initialized", False):
        return

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    env_pool = []
    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            payload = {
                "task_id": GAMES_TO_TASK_ID_RANGE[SELECTED_GAME][0],
                "seed": 42,
                "opponent": "mcts",
                "mcts_max_simulations": 25,
                "mcts_num_rollouts": 1,
            }
            res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}")

    rollout_warmup_rollouts = (
        trainer.args.rollout_warmup_rollouts
        if getattr(trainer.args, "rollout_warmup_rollouts", None) is not None
        else trainer.args.rollouts_per_stage
    )
    mcts_warmup_optimizer_steps = getattr(trainer.args, "mcts_warmup_optimizer_steps", None)
    hint_decay_optimizer_steps = 100

    curriculum = CurriculumScheduler(
        initial_max_turn=trainer.args.initial_max_turn,
        final_max_turn=8,  # Leduc max ~8 actions per game
        rollouts_per_stage=trainer.args.rollouts_per_stage,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        hint_decay_optimizer_steps=hint_decay_optimizer_steps,
        warmup_rollouts=rollout_warmup_rollouts,
        mcts_warmup_optimizer_steps=mcts_warmup_optimizer_steps,
        initial_mcts_sims=25,
        final_mcts_sims=25,
    )

    fn.rank = rank
    fn.env_pool = env_pool
    fn.num_servers = len(env_pool)
    fn.initialized = True
    fn.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
    fn.generation_semaphore = Semaphore(1)
    fn.curriculum = curriculum

    print(
        f"[CURRICULUM] Initialized (leduc_poker) with "
        f"initial_max_turn={trainer.args.initial_max_turn}, final_max_turn=8, "
        f"rollouts_per_stage={trainer.args.rollouts_per_stage}, "
        f"rollout_warmup_rollouts={rollout_warmup_rollouts}, "
        f"hint_decay_optimizer_steps={hint_decay_optimizer_steps}, "
        f"mcts_warmup_optimizer_steps={mcts_warmup_optimizer_steps}, "
        f"mcts_sims=25->25 (constant)"
    )


# ---------------------------------------------------------------------------
# rollout_last_prompt_and_completion_parallelized_curriculum
# ---------------------------------------------------------------------------

def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 8,
) -> dict[str, list]:
    """Parallelized rollout (last prompt+completion variant) for Leduc Poker."""

    _init_static(rollout_last_prompt_and_completion_parallelized_curriculum, trainer)

    fn = rollout_last_prompt_and_completion_parallelized_curriculum
    rank = fn.rank
    env_pool = fn.env_pool
    num_servers = fn.num_servers
    curriculum = fn.curriculum
    tokenizer = trainer.processing_class

    total_rollouts = curriculum.total_rollouts
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob(current_optimizer_step)
    current_mcts_sims = curriculum.get_mcts_sims(current_optimizer_step)
    print(
        f"[CURRICULUM] Rollout {total_rollouts}, step {current_optimizer_step}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        env_endpoint = env_pool[server_idx]["base_url"]

        invalid_count = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        game_state_history: list[GameState] = []
        rewards = []
        calculator = RewardCalculator()
        use_hints = random.random() < current_hint_prob

        # --- Reset ---
        payload = {
            "task_id": game_id,
            "seed": random.randint(0, 2**31 - 1),
            "opponent": "mcts",
            "mcts_max_simulations": current_mcts_sims,
            "mcts_num_rollouts": 1,
        }
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            result_block = reset_res.json()["result"]
            episode_id = result_block.get("episode_id", "")
            raw_obs = result_block.get("observation", "")
            formatted_obs = extract_and_format_observation(raw_obs)
            initial_state = parse_game_state(formatted_obs)
            game_state_history.append(initial_state)
        except Exception as e:
            print(f"Failed to reset (Game {game_id}): {e}")
            return index, None

        # --- Build messages ---
        system_prompt = LEDUC_SYSTEM_PROMPT
        if use_hints:
            system_prompt += LEDUC_HINT_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": formatted_obs},
        ]

        prompt_ids = []
        completion_ids = []
        logprobs = []

        # --- Interaction Loop ---
        while not done and turn_number < current_max_turn:
            with fn.generation_semaphore:
                rollout_outputs = generate_rollout_completions(
                    trainer, prompts=[messages], as_chat=True
                )[0]

            prompt_ids     = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs       = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": completion_text})

            action_to_send = extract_action_id(completion_text)

            # --- Step ---
            try:
                step_res = requests.post(
                    f"{env_endpoint}/step",
                    json={"action": action_to_send, "episode_id": episode_id},
                    timeout=TIMEOUT,
                )
                step_res.raise_for_status()
                step_block = step_res.json()["result"]
                raw_obs         = step_block.get("observation", "")
                formatted_obs   = extract_and_format_observation(raw_obs)
                step_reward     = step_block.get("reward", 0)
                done            = step_block.get("done", False)
            except Exception as e:
                print(f"Step failed: {e}")
                step_reward  = -0.01
                done         = False
                invalid_count += 1

            is_invalid = False
            if "Nothing happens" in formatted_obs or "Invalid" in formatted_obs:
                invalid_count += 1
                is_invalid = True

            if done:
                final_reward = step_reward
            else:
                messages.append({"role": "user", "content": formatted_obs})
                if not is_invalid:
                    try:
                        game_state = parse_game_state(formatted_obs)
                        game_state_history.append(game_state)
                    except Exception:
                        pass

            immediate_reward = calculator.calculate_step_reward(
                game_state_history, action_to_send, 0.0, is_invalid=is_invalid
            )
            rewards.append(immediate_reward)
            turn_number += 1

        initial_st = game_state_history[0] if game_state_history else None
        final_st   = game_state_history[-1] if game_state_history else None
        episode_reward = calculator.calculate_episode_reward(
            rewards, final_reward, done, initial_st, final_st,
            all_states=game_state_history,
        )

        print(
            f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{episode_reward:6.2f} EnvR:{final_reward:5.1f} Inv:{invalid_count}]"
        )

        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         episode_reward,
            "final_score":    final_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    futures = [
        fn.thread_pool.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else {
            "prompt_ids": [1], "completion_ids": [1],
            "logprobs": [1.0], "reward": 0.0, "final_score": 0.0,
        }

    curriculum.step(len(prompts))
    list_results = [r for r in results if r is not None]

    finished  = sum(1 for r in list_results if r["final_score"] != 0)
    wins      = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}")

    return {
        "prompt_ids":    [r["prompt_ids"]     for r in list_results],
        "completion_ids":[r["completion_ids"] for r in list_results],
        "logprobs":      [r["logprobs"]        for r in list_results],
        "env_rewards":   [r["reward"]          for r in list_results],
    }


# ---------------------------------------------------------------------------
# rollout_full_prompt_and_completion_parallelized_curriculum
# ---------------------------------------------------------------------------

def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 8,
) -> dict[str, list]:
    """
    Parallelized rollout (full prompt+completion with action masking) for Leduc Poker.
    """
    MAX_EPISODE_TOKENS = 16384
    MAX_PROMPT_LEN     = 16384 - 256

    _init_static(rollout_full_prompt_and_completion_parallelized_curriculum, trainer)

    fn = rollout_full_prompt_and_completion_parallelized_curriculum
    rank = fn.rank
    env_pool = fn.env_pool
    num_servers = fn.num_servers
    curriculum = fn.curriculum
    tokenizer = trainer.processing_class

    total_rollouts = curriculum.total_rollouts
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob(current_optimizer_step)
    current_mcts_sims = curriculum.get_mcts_sims(current_optimizer_step)
    print(
        f"[CURRICULUM] Rollout {total_rollouts}, step {current_optimizer_step}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id    = int(prompt)
        server_idx = (index + rank) % num_servers
        env_endpoint = env_pool[server_idx]["base_url"]

        episode_prompt_ids:    list[int]   = []
        episode_completion_ids: list[int]  = []
        episode_logprobs:      list[float] = []
        episode_action_mask:   list[int]   = []
        prev_full_ids: Optional[list[int]] = None

        invalid_count = 0
        done          = False
        final_reward  = 0.0
        turn_number   = 0
        game_state_history: list[GameState] = []
        rewards: list[float] = []
        calculator = RewardCalculator()
        use_hints  = random.random() < current_hint_prob

        # --- Reset ---
        payload = {
            "task_id": game_id,
            "seed": random.randint(0, 2**31 - 1),
            "opponent": "mcts",
            "mcts_max_simulations": current_mcts_sims,
            "mcts_num_rollouts": 1,
        }
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            result_block = reset_res.json()["result"]
            episode_id   = result_block.get("episode_id", "")
            raw_obs      = result_block.get("observation", "")
            formatted_obs = extract_and_format_observation(raw_obs)
            initial_state = parse_game_state(formatted_obs)
            game_state_history.append(initial_state)
        except Exception as e:
            print(f"Failed to reset (Game {game_id}): {e}")
            return index, None

        # --- Build messages ---
        system_prompt = LEDUC_SYSTEM_PROMPT
        if use_hints:
            system_prompt += LEDUC_HINT_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": formatted_obs},
        ]

        # --- Interaction Loop ---
        while not done and turn_number < current_max_turn:
            with fn.generation_semaphore:
                rollout_outputs = generate_rollout_completions(
                    trainer, prompts=[messages], as_chat=True
                )[0]

            prompt_ids     = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs       = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            action_to_send  = extract_action_id(completion_text)

            # --- Track full prompt/completion IDs ---
            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                done = True
                break

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

            messages.append({"role": "assistant", "content": completion_text})

            # --- Step ---
            try:
                step_res = requests.post(
                    f"{env_endpoint}/step",
                    json={"action": action_to_send, "episode_id": episode_id},
                    timeout=TIMEOUT,
                )
                step_res.raise_for_status()
                step_block    = step_res.json()["result"]
                raw_obs       = step_block.get("observation", "")
                formatted_obs = extract_and_format_observation(raw_obs)
                step_reward   = step_block.get("reward", 0)
                done          = step_block.get("done", False)
            except Exception as e:
                print(f"Step failed: {e}")
                step_reward   = -0.01
                done          = False
                invalid_count += 1

            is_invalid = False
            if "Nothing happens" in formatted_obs or "Invalid" in formatted_obs:
                invalid_count += 1
                is_invalid = True

            if done:
                final_reward = step_reward
                messages.append({"role": "user", "content": formatted_obs})
            else:
                messages.append({"role": "user", "content": formatted_obs})
                if not is_invalid:
                    try:
                        game_state = parse_game_state(formatted_obs)
                        game_state_history.append(game_state)
                    except Exception as e:
                        print(f"Failed to parse game state: {e}")

            immediate_reward = calculator.calculate_step_reward(
                game_state_history, action_to_send, 0.0, is_invalid=is_invalid
            )
            rewards.append(immediate_reward)
            turn_number += 1

        initial_st = game_state_history[0] if game_state_history else None
        final_st   = game_state_history[-1] if game_state_history else None
        episode_reward = calculator.calculate_episode_reward(
            rewards, final_reward, done, initial_st, final_st,
            all_states=game_state_history,
        )

        print(
            f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{episode_reward:6.2f} EnvR:{final_reward:5.1f} Inv:{invalid_count}]"
        )

        # Truncate if needed
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(
                f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens "
                f"({len(episode_completion_ids)}), truncating"
            )
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:MAX_EPISODE_TOKENS]

        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         episode_reward,
            "final_score":    final_reward,
        }

    # --- Execute in parallel ---
    results = [None] * len(prompts)
    futures = [
        fn.thread_pool.submit(run_single_prompt, i, p)
        for i, p in enumerate(prompts)
    ]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else {
            "prompt_ids": [1], "completion_ids": [1],
            "action_mask": [0], "logprobs": [1.0],
            "reward": 0.0, "final_score": 0.0,
        }

    curriculum.step(len(prompts))
    list_results = [r for r in results if r is not None]

    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    wins       = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}")

    return {
        "prompt_ids":    [r["prompt_ids"]     for r in list_results],
        "completion_ids":[r["completion_ids"] for r in list_results],
        "action_mask":   [r["action_mask"]    for r in list_results],
        "logprobs":      [r["logprobs"]        for r in list_results],
        "env_rewards":   [r["reward"]          for r in list_results],
    }


# ---------------------------------------------------------------------------
# Reward Function (used by TRL trainer)
# ---------------------------------------------------------------------------

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)