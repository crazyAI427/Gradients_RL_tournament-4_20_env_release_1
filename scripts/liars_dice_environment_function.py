"""
leduc_poker_environment_function.py  —  MERGED for Release-1
  Drop-in replacement for: scripts/leduc_poker_environment_function.py
  (no shared_env.py dependency — all helpers are inlined)
================================================
Sources and their unique contributions
---------------------------------------

Release-1  (Gradients_RL_tournament-4_20_env_release_1)
  • Chip-improvement reward component (CHIP_WEIGHT)
  • PAIR_BONUS / HIGHCARD_WIN_BONUS on terminal
  • INVALID_TOTAL_CLIP — caps accumulated invalid penalties at -0.3
  • remove_reasoning_tags() — strips <think>/<reasoning>/etc. from model output
  • extract_action_id() — robust numeric-action extractor (regex fallback)
  • random seed per episode (random.randint) for env diversity
  • MCTS sim curriculum: get_mcts_sims(optimizer_step) warm-up schedule
  • hint_decay driven by optimizer_step (not just rollout count)
  • global_step awareness via trainer.state
  • Wins counter in batch log
  • rollout_warmup_rollouts separate from rollouts_per_stage

Boss-repo (boss-repo-init)
  • Rich named SIGNALS dict — per-action shaping with semantic labels
  • GameState.hand_strength, is_strong, is_weak, raises_this_round, opp_last_action
  • discounted_return aggregation with gamma
  • Module-level _state dict (single init path, no per-function attributes)

Release-6  (Gradients_RL_tournament-4_20_env_release_6)
  • terminal_weight = 10.0 — win/loss dominates shaping noise
  • NORMALIZE_REWARDS — length-invariant intermediate shaping:
      G = (Σ γ^(T-1-i) * s_i) / T  +  terminal_reward
  • Separate step_scores / terminal_reward tracking
  • shared_env.py imports (GAMES_TO_TASK_ID_RANGE, CurriculumScheduler,
    init_env_pool, rollout_reward_func)

New additions (combined logic)
  • Pot-odds call bonus: reward calling when equity_estimate ≥ (1 − pot_odds)
  • Q + public K raise bonus in R2 (second-best hand)
  • Positional raise bonus: P1 raising K in R1 (info-advantage aggression)
  • Raise-cap call bonus: reward calling when 2-raise cap is hit
  • PAIR_BONUS and HIGHCARD_WIN_BONUS applied inside terminal_reward
  • chip_component from R1 added to terminal_reward bucket (normalized)
"""

import functools
import os
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass
from threading import Semaphore
from typing import Optional

import requests
from trl.experimental.openenv import generate_rollout_completions

# ---------------------------------------------------------------------------
# Inline definitions (Release-1 has no shared_env.py)
# ---------------------------------------------------------------------------

GAMES_TO_TASK_ID_RANGE: dict[str, tuple[int, int]] = {
    "goofspiel":   (0,         99_999_999),
    "liars_dice":  (100000000, 199_999_999),
    "leduc_poker": (200000000, 299_999_999),
    "gin_rummy":   (300000000, 399_999_999),
    "othello":     (400000000, 499_999_999),
    "backgammon":  (500000000, 599_999_999),
    "hex":         (600000000, 699_999_999),
    "clobber":     (700000000, 799_999_999),
}


def init_env_pool(reset_payload: dict):
    """Initialise environment server pool (inline, no shared_env dependency)."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")
    env_pool: list[dict] = []
    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Connecting to server {idx}: {base_url}")
            res = requests.post(f"{base_url}/reset", json=reset_payload, timeout=300)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as exc:
            raise RuntimeError(f"Failed to init server {base_url}: {exc}") from exc
    thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
    generation_semaphore = Semaphore(1)
    return rank, env_pool, len(env_pool), thread_pool, generation_semaphore


def rollout_reward_func(completions, **kwargs):
    """Generic reward passthrough (inline for Release-1 compatibility)."""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME      = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN     = 16384 - 256
_TIMEOUT            = 2400
_MAX_TURNS          = 10           # hard episode cap

# ── Reward constants (R1 + Release-6) ──────────────────────────────────────
TERMINAL_WIN_REWARD  =  1.0
TERMINAL_LOSS_REWARD = -1.0
PAIR_BONUS           =  0.15   # R1: bonus when winning hand contains a pair
HIGHCARD_WIN_BONUS   =  0.05   # R1: bonus for winning with high card
CHIP_WEIGHT          =  0.15   # R1 (scaled): chip-improvement fraction of reward
INVALID_PENALTY      = -0.1    # per invalid step
INVALID_TOTAL_CLIP   = -0.3    # R1: hard floor on accumulated invalid penalty

# ── Normalize intermediate rewards by episode length (Release-6) ───────────
NORMALIZE_REWARDS = True


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits \u00d7 (num_players + 1) ranks. "
    "For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), "
    "Call/Check (match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n"
    "\n# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> fold": respond "0"\n'
    '- For action "1 -> call": respond "1"\n'
    '- For action "2 -> raise": respond "2"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "Round 1:\n"
    "- Hold K or Q \u2192 call a raise; raise first if unchallenged.\n"
    "- Hold J \u2192 fold against a raise; check if unchallenged.\n\n"
    "Round 2 (public card revealed):\n"
    "- You have a PAIR \u2192 raise aggressively; never fold.\n"
    "- You have K (no pair) \u2192 raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is K \u2192 raise; call if opponent raises.\n"
    "- You have Q (no pair), public card is J \u2192 check; fold if opponent raises.\n"
    "- You have J (no pair) \u2192 check; fold if opponent raises.\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
)


# ---------------------------------------------------------------------------
# Card utilities  (R1)
# ---------------------------------------------------------------------------

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


def _card_rank(card_name: str) -> int:
    if not card_name:
        return 0
    return _CARD_RANK.get(card_name[0].upper(), 0)


def _is_pair(private: str, public: str) -> bool:
    if not private or not public:
        return False
    return private[0].upper() == public[0].upper()


# ---------------------------------------------------------------------------
# GameState  (boss rich properties + R1 chip tracking)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Unified Leduc Poker game state.

    Combines:
    - Boss:    player_id, opp_last_action, raises_this_round, hand_strength,
               pot_odds_ratio, equity_estimate, can_raise, max_raises_reached
    - R1:      my_chips, opp_chips, phase, chip-delta computation
    - Merged:  has_pair() callable + has_pair bool from boss parser
    """
    # Core fields (populated by parser)
    player_id:         int
    private_card:      str           # e.g. "K\u2660"
    private_card_rank: int           # J=1, Q=2, K=3
    public_card:       Optional[str] # None in round 1
    public_card_rank:  Optional[int]
    pair:              bool          # True when Hand: Pair in observation
    round:             int           # 1 or 2
    pot:               int
    my_chips:          int           # alias: our_chips
    opp_chips:         int
    r1_betting:        list[str]
    r2_betting:        list[str]
    legal_actions:     dict[int, str]

    # ── Derived helpers ─────────────────────────────────────────────────────

    def has_pair(self) -> bool:
        """Callable form for R1 compatibility."""
        return self.pair

    @property
    def our_invested(self) -> int:
        return 100 - self.my_chips

    @property
    def opp_invested(self) -> int:
        return 100 - self.opp_chips

    @property
    def current_round_betting(self) -> list[str]:
        return self.r1_betting if self.round == 1 else self.r2_betting

    @property
    def raises_this_round(self) -> int:
        return self.current_round_betting.count("Raise")

    @property
    def opp_last_action(self) -> Optional[str]:
        betting = self.current_round_betting
        return betting[-1] if betting else None

    @property
    def opp_raised_this_round(self) -> bool:
        return "Raise" in self.current_round_betting

    @property
    def can_raise(self) -> bool:
        return 2 in self.legal_actions

    @property
    def can_fold(self) -> bool:
        return 0 in self.legal_actions

    @property
    def max_raises_reached(self) -> bool:
        """True when no more raises are legal this round (cap = 2)."""
        return not self.can_raise and self.raises_this_round >= 2

    @property
    def hand_strength(self) -> int:
        """
        1–4 scale: Pair=4, K=3, Q=2, J=1
        """
        if self.pair:
            return 4
        return self.private_card_rank

    @property
    def is_strong(self) -> bool:
        return self.hand_strength >= 3

    @property
    def is_weak(self) -> bool:
        return self.hand_strength == 1

    @property
    def pot_odds_ratio(self) -> float:
        """pot / (pot + call_cost); approximates required equity to call."""
        call_cost = max(self.opp_invested - self.our_invested, 0)
        if call_cost == 0:
            return 1.0
        return self.pot / (self.pot + call_cost)

    @property
    def equity_estimate(self) -> float:
        """Hand-strength-based equity (0.0–1.0)."""
        return {4: 0.85, 3: 0.65, 2: 0.40, 1: 0.20}.get(self.hand_strength, 0.30)

    @property
    def call_is_profitable(self) -> bool:
        return self.equity_estimate >= (1.0 - self.pot_odds_ratio)

    # R1 compat alias
    @property
    def total_chips(self) -> int:
        return self.my_chips


# ---------------------------------------------------------------------------
# Observation parser
# ---------------------------------------------------------------------------

def _format_observation(raw: str) -> str:
    """
    Reformat server observation to eval-framework format.
    Strips game-rules preamble, normalises indents and prompt wording.
    (Boss impl — handles both server formats)
    """
    player_match = re.search(r"You are Player (\d+)\.", raw)
    player_line  = f"You are Player {player_match.group(1)}." if player_match else ""

    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw

    body = raw[state_start:]
    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body

    state_block   = body[:legal_start].rstrip()
    actions_block = body[legal_start:]
    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)
    actions_block = actions_block.replace(
        "Your choice (action ID only):", "Your choice (ID only):"
    )

    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n\n".join(parts)


def parse_game_state(obs: str) -> Optional[GameState]:
    """
    Parse a Leduc Poker observation into a GameState.
    Returns None if observation cannot be parsed (invalid / empty).
    """
    if not obs or "Current State:" not in obs:
        return None

    def _find(pattern, default=None):
        m = re.search(pattern, obs)
        return m.group(1) if m else default

    pid_str = _find(r"You are Player (\d+)\.")
    if pid_str is None:
        return None
    player_id = int(pid_str)

    private_card = _find(r"Your card:\s*(\S+)")
    if private_card is None:
        return None
    private_card_rank = _card_rank(private_card)

    pub_raw          = _find(r"Public card:\s*(\S+)")
    public_card      = pub_raw
    public_card_rank = _card_rank(pub_raw) if pub_raw else None

    pair = "Hand: Pair" in obs

    round_str = _find(r"Current round:\s*(\d+)/\d+", "1")
    round_    = int(round_str)

    pot       = int(_find(r"Pot size:\s*(\d+)", "0"))
    my_chips  = int(_find(r"Your chips:\s*(\d+)", "100"))
    opp_chips = int(_find(r"Opponent chips:\s*(\d+)", "100"))

    def _parse_betting(label: str) -> list[str]:
        m = re.search(rf"{label} betting:\s*(.+)", obs)
        if not m:
            return []
        return [a.strip() for a in m.group(1).split(",")]

    r1_betting = _parse_betting("Round 1")
    r2_betting = _parse_betting("Round 2")

    legal_actions: dict[int, str] = {}
    for m in re.finditer(r"^(\d+)\s*->\s*(.+)$", obs, re.MULTILINE):
        legal_actions[int(m.group(1))] = m.group(2).strip()

    return GameState(
        player_id=player_id,
        private_card=private_card,
        private_card_rank=private_card_rank,
        public_card=public_card,
        public_card_rank=public_card_rank,
        pair=pair,
        round=round_,
        pot=pot,
        my_chips=my_chips,
        opp_chips=opp_chips,
        r1_betting=r1_betting,
        r2_betting=r2_betting,
        legal_actions=legal_actions,
    )


# ---------------------------------------------------------------------------
# Action extraction  (R1 — reasoning-tag aware)
# ---------------------------------------------------------------------------

_REASONING_TAG_PAIRS = [
    ("think",      "think"),
    ("thinking",   "thinking"),
    ("reasoning",  "reasoning"),
    ("thought",    "thought"),
    ("reflection", "reflection"),
]


def _remove_reasoning_tags(text: str) -> str:
    """Strip <think>…</think> and similar reasoning blocks from model output."""
    cleaned = text
    for tag, close in _REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag}>.*?</{close}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    return re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned).strip()


def _extract_action_id(completion_text: str) -> str:
    """
    Robustly extract a numeric action ID from model output.
    - Strips reasoning tags (R1)
    - Handles 'Action: N' format (boss)
    - Regex fallback to first integer (R1)
    """
    cleaned = _remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    match = re.search(r"-?\d+", cleaned)
    return match.group(0) if match else cleaned.strip()


# ---------------------------------------------------------------------------
# Reward Calculator  (all three repos merged)
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Unified reward calculator for Leduc Poker.

    Per-step shaping (boss named SIGNALS + new combined signals):
      • Fold penalties for strong hands (pair, K, Q/K vs raise)
      • Fold bonuses for weak hands under pressure (J, Q+pubJ)
      • Raise bonuses (pair R2, K R2, Q+pubK R2, K R1 positional)
      • Call bonuses (Q/K vs raise, pot-odds profitable, raise-cap)
      • Invalid penalty per step

    Episode aggregation (Release-6 normalized + R1 terminal enrichment):
      • terminal_weight = 10.0  — win/loss dominates shaping
      • PAIR_BONUS / HIGHCARD_WIN_BONUS on wins  (R1)
      • chip_component = chip delta * CHIP_WEIGHT  (R1, added to terminal)
      • INVALID_TOTAL_CLIP caps accumulated invalid penalties at -0.3  (R1)
      • NORMALIZE_REWARDS: G = (Σ γ^(T-1-i)*s_i)/T + terminal_reward  (R6)
    """

    # ── Named shaping signals (boss + new) ───────────────────────────────
    SIGNALS: dict[str, float] = {
        # Bad folds
        "fold_pair":               -2.0,
        "fold_k":                  -1.5,
        "fold_kq_r1_raise":        -1.5,
        "fold_q_pubk_raise":       -0.5,
        # Good folds
        "fold_j_r1_raise":         +0.3,
        "fold_j_r2_raise":         +0.2,
        "fold_q_pubj_raise":       +0.2,
        # Raise bonuses
        "raise_pair_r2":           +0.4,
        "raise_k_r2":              +0.2,
        "raise_q_pubk_r2":         +0.2,   # NEW: Q + pub K = 2nd-best hand
        "raise_k_r1_p1":           +0.15,  # NEW: positional aggression (P1)
        # Call / check bonuses
        "call_kq_r1_raise":        +0.2,
        "call_pot_odds_profitable": +0.15,  # NEW: pot-odds justified call
        "call_when_raises_maxed":   +0.1,   # NEW: raise cap hit; call is correct
    }

    def __init__(self, terminal_weight: float = 10.0, gamma: float = 0.9):
        self.terminal_weight = terminal_weight
        self.gamma           = gamma

    # ── Per-step shaping ─────────────────────────────────────────────────
    def calculate_step_reward(
        self,
        gs: Optional[GameState],
        action_str: str,
        is_invalid: bool = False,
    ) -> float:
        """
        Return per-step shaping score (no terminal component).
        Invalid actions return INVALID_PENALTY directly.
        """
        if is_invalid:
            return INVALID_PENALTY

        if gs is None:
            return 0.0

        reward = 0.0
        pub    = gs.public_card_rank or 0

        if action_str == "Fold":
            if gs.pair:
                reward += self.SIGNALS["fold_pair"]
            elif gs.private_card_rank == 3:
                reward += self.SIGNALS["fold_k"]
            elif gs.round == 1 and gs.opp_last_action == "Raise":
                if gs.private_card_rank >= 2:
                    reward += self.SIGNALS["fold_kq_r1_raise"]
                else:
                    reward += self.SIGNALS["fold_j_r1_raise"]
            elif gs.round == 2 and gs.opp_last_action == "Raise":
                if gs.private_card_rank == 1:
                    reward += self.SIGNALS["fold_j_r2_raise"]
                elif gs.private_card_rank == 2 and pub == 1:
                    reward += self.SIGNALS["fold_q_pubj_raise"]
                elif gs.private_card_rank == 2 and pub == 3:
                    reward += self.SIGNALS["fold_q_pubk_raise"]

        elif action_str == "Raise":
            if gs.round == 2 and gs.pair:
                reward += self.SIGNALS["raise_pair_r2"]
            elif gs.round == 2 and gs.private_card_rank == 3 and not gs.pair:
                reward += self.SIGNALS["raise_k_r2"]
            elif gs.round == 2 and gs.private_card_rank == 2 and pub == 3:
                reward += self.SIGNALS["raise_q_pubk_r2"]
            elif gs.round == 1 and gs.player_id == 1 and gs.private_card_rank == 3:
                reward += self.SIGNALS["raise_k_r1_p1"]

        elif action_str in ("Call", "Check"):
            if gs.round == 1 and gs.opp_last_action == "Raise" and gs.private_card_rank >= 2:
                reward += self.SIGNALS["call_kq_r1_raise"]
            if gs.opp_last_action == "Raise" and gs.call_is_profitable:
                reward += self.SIGNALS["call_pot_odds_profitable"]
            if gs.max_raises_reached:
                reward += self.SIGNALS["call_when_raises_maxed"]

        return reward

    # ── Episode aggregation ───────────────────────────────────────────────
    def calculate_terminal_reward(
        self,
        env_reward: float,
        done: bool,
        initial_state: Optional[GameState],
        final_state:   Optional[GameState],
        invalid_penalties: float,
    ) -> float:
        """
        Compute the terminal component of the training return.

        Includes (in priority order):
          1. Win/loss * terminal_weight           (Release-6)
          2. PAIR_BONUS / HIGHCARD_WIN_BONUS      (R1)
          3. Chip improvement component            (R1, scaled)
          4. Accumulated invalid penalties clipped (R1)
        """
        terminal = 0.0

        if done:
            if env_reward > 0.5:
                terminal = TERMINAL_WIN_REWARD * self.terminal_weight
                if final_state and final_state.pair:
                    terminal += PAIR_BONUS
                else:
                    terminal += HIGHCARD_WIN_BONUS
            else:
                terminal = TERMINAL_LOSS_REWARD * self.terminal_weight
        elif final_state:
            # Partial progress signal when episode times out
            terminal = -final_state.hand_strength / 40.0

        # Chip improvement (R1) — added to terminal bucket
        if initial_state and final_state and initial_state.my_chips > 0:
            chip_delta     = (final_state.my_chips - initial_state.my_chips) / initial_state.my_chips
            terminal      += chip_delta * CHIP_WEIGHT

        # Clip accumulated invalid penalties (R1)
        clipped_inv = max(invalid_penalties, INVALID_TOTAL_CLIP)
        terminal   += clipped_inv

        return terminal

    def calculate_discounted_return(
        self,
        step_scores:     list[float],
        terminal_reward: float = 0.0,
    ) -> float:
        """
        Compute training return.

        NORMALIZE_REWARDS = True  (Release-6):
            G = (Σ γ^(T-1-i) * s_i) / T  +  terminal_reward
            Length-invariant: short and long episodes contribute equally.

        NORMALIZE_REWARDS = False  (legacy boss):
            G = Σ γ^(T-1-i) * r_i
        """
        if not NORMALIZE_REWARDS:
            if not step_scores:
                return terminal_reward
            T = len(step_scores)
            return (
                sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(step_scores))
                + terminal_reward
            )

        if not step_scores:
            return terminal_reward
        T = len(step_scores)
        discounted_sum = sum(
            self.gamma ** (T - 1 - i) * s for i, s in enumerate(step_scores)
        )
        return discounted_sum / T + terminal_reward


# ---------------------------------------------------------------------------
# Module-level state  (boss pattern — single shared dict per process)
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Build CurriculumScheduler from trainer args (Release-6 style)."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=getattr(args, "rollout_warmup_rollouts", args.rollouts_per_stage),
    )


def _ensure_initialized(trainer) -> None:
    """One-time env-pool and curriculum setup (no-op on subsequent calls)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed":    42,
        "opponent": "mcts",
        "mcts_max_simulations": 25,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)
    curriculum = _curriculum_factory(trainer.args)

    print(
        f"[CURRICULUM] Initialized (leduc_poker): "
        f"initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={_MAX_TURNS}, "
        f"rollouts_per_stage={trainer.args.rollouts_per_stage}"
    )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
    )


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt:      bool,
    env_pool:             list[dict],
    num_servers:          int,
    rank:                 int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_hint_prob:    float,
    current_mcts_sims:    int,
) -> tuple[int, Optional[dict]]:
    """
    Run one Leduc Poker episode.

    Reward flow (merged):
      step_scores       — per-turn shaping (no terminal); normalized by T
      terminal_reward   — win/loss * 10 + PAIR_BONUS + chip_delta (R1+R6)
      invalid_penalties — accumulated separately, clipped at INVALID_TOTAL_CLIP
    """
    game_id      = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation
    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: Optional[list[int]]  = None

    # Last-turn state (overwritten each turn)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done              = False
    final_reward      = 0.0
    turn_number       = 0
    invalid_count     = 0
    invalid_penalties = 0.0       # accumulates INVALID_PENALTY per bad action
    use_hints         = random.random() < current_hint_prob

    calculator         = RewardCalculator()
    step_scores:       list[float] = []
    game_state_history: list[GameState] = []

    # ── Reset ──────────────────────────────────────────────────────────────
    reset_payload = {
        "task_id":              game_id,
        "seed":                 random.randint(0, 2**31 - 1),   # R1: diverse seeds
        "opponent":             "mcts",
        "mcts_max_simulations": current_mcts_sims,              # R1: curriculum sims
        "mcts_num_rollouts":    1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id   = result_block.get("episode_id", "")
        observation  = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": observation},
    ]

    # ── Interaction loop ───────────────────────────────────────────────────
    while not done and turn_number < _MAX_TURNS:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(
                trainer, prompts=[messages], as_chat=True
            )[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # ── Full-prompt token accumulation ─────────────────────────────────
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending early"
                )
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids      = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask."
                    )
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # ── Parse action & step ─────────────────────────────────────────────
        action_to_send = _extract_action_id(completion_text)  # R1 robust extractor
        prev_gs = game_state_history[-1] if game_state_history else None

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            step_reward = step_block.get("reward", 0)
            done        = step_block.get("done", False)
            if not done:
                gs = parse_game_state(observation)
                if gs is not None:
                    game_state_history.append(gs)
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation  = ""
            step_reward  = 0
            done         = False
            invalid_count += 1

        is_invalid = "Nothing happens" in observation or "Invalid" in observation
        if is_invalid:
            invalid_count += 1

        if done:
            final_reward = step_reward

        # ── Per-step shaping ────────────────────────────────────────────────
        try:
            action_str = (
                prev_gs.legal_actions.get(int(action_to_send.strip()), "")
                if prev_gs else ""
            )
        except (ValueError, AttributeError):
            action_str = ""

        step_score = calculator.calculate_step_reward(prev_gs, action_str, is_invalid=is_invalid)
        step_scores.append(step_score)
        if is_invalid:
            invalid_penalties += INVALID_PENALTY

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    # ── Episode reward ──────────────────────────────────────────────────────
    initial_st = game_state_history[0]  if game_state_history else None
    final_st   = game_state_history[-1] if game_state_history else None

    terminal_reward = calculator.calculate_terminal_reward(
        env_reward=final_reward,
        done=done,
        initial_state=initial_st,
        final_state=final_st,
        invalid_penalties=invalid_penalties,
    )
    train_reward = calculator.calculate_discounted_return(
        step_scores=step_scores,
        terminal_reward=terminal_reward,
    )

    print(
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{} | MCTS:{:>2d} | "
        "EnvR:{:>6.2f} | TrainR:{:>7.3f} | Inv:{} | Norm:{}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints),
            current_mcts_sims, final_reward, train_reward,
            invalid_count, int(NORMALIZE_REWARDS),
        )
    )

    # ── Build result ────────────────────────────────────────────────────────
    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
        }
    else:
        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
        }


# ---------------------------------------------------------------------------
# Shared dispatch  (boss + Release-6 pattern)
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    """Common parallelisation + aggregation for both rollout variants."""
    _ensure_initialized(trainer)

    curriculum         = _state["curriculum"]
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_hint_prob  = curriculum.get_hint_prob()               # rollout-count decay
    current_mcts_sims  = getattr(curriculum, "get_mcts_sims",     # R1: optional mcts
                                  lambda *_: 25)(current_optimizer_step)

    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}, "
        f"step {current_optimizer_step}: "
        f"hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_hint_prob=current_hint_prob,
        current_mcts_sims=current_mcts_sims,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0],
         "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1],
         "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    wins       = sum(1 for r in list_results if r["final_score"] > 0.5)   # R1
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(
        f"[BATCH] Finished:{finished}/{len(list_results)} "
        f"Wins:{wins} AvgReturn:{avg_return:.3f}"                          # R1 wins
    )

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)