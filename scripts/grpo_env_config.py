from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    disable_action_mask,
    get_use_vllm,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_grpo_lr
allow_find_lk_lr = False

GRPO_CONFIG = {
    #can ignore in env tourn
    "0_1_b": { 
        "lr": 2e-4,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 6,
        "vllm_gpu_memory_utilization": 0.4,
        "use_lora": True,
        "beta": 0.02,
        "num_generations": 4,
        "initial_max_turn": 1,
        "rollouts_per_stage": 1280,  # Reduced for more frequent curriculum progression
    },
    #can ignore in env tourn
    "1_2_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 3,
        "gradient_accumulation_steps": 12,
        "vllm_gpu_memory_utilization": 0.4,
        "beta": 0.04,
        "num_generations": 4,
        "rollouts_per_stage": 1280,
    },
    
    "2_4_b": {
        "lr": 1e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "vllm_gpu_memory_utilization": 0.3,
        "use_lora": True,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1280,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    "2_4_b_qwen": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "vllm_gpu_memory_utilization": 0.3,
        "use_lora": True,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1280,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    # only qwen3-4B-instruct-2507
    "4_5_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,  # Reduced for Gin Rummy
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1280,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    # can ignore in env tourn
    "5_6_b": {
        "lr": 3e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,  # Reduced for Gin Rummy
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1280,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },

    "6_9_b": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,  # Reduced for Gin Rummy (longer episodes = more KV cache)
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1024,  # Increased from 768 for better curriculum
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    # Qwen2-7B-Instruct: medium-high LR tolerance.

    "6_9_b_qwen2": {
        "lr": 2e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1024,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },

    "6_9_b_qwen2_5": {
        "lr": 4e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1024,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },

    "6_9_b_mistral": {
        "lr": 1e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1024,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },

    "6_9_b_mistral_effbs8": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.35,
        "beta": 0.01,
        "num_generations": 8,
        "rollouts_per_stage": 1024,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    # can ignore in env tourn
    "9_12_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "beta": 0.01,
        "rollout_warmup_rollouts": 0,
        "mcts_warmup_optimizer_steps": 20,
    },
    # can ignore in env tourn
    "12_15_b": {
        "lr": 3e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.8,
        "beta": 0.01,
    },
    "15_20_b": {
        "lr": 1e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
        "beta": 0.01,
    },
    "20_40_b": {
        "lr": 5e-7,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
        "use_4bit": True,
        "beta": 0.01,
    },
    "40_80_b": {
        "lr": 3e-7,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.7,
        "use_vllm": False,
        "use_4bit": True,
        "beta": 0.01,
    },
}

for key in GRPO_CONFIG:
    GRPO_CONFIG[key]["label"] = key


def get_grpo_config(param_nums: int) -> dict:
    if param_nums < 1_000_000_000:
        return GRPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return GRPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return GRPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return GRPO_CONFIG["4_5_b"]
    elif param_nums < 6_000_000_000:
        return GRPO_CONFIG["5_6_b"]
    elif param_nums < 9_000_000_000:
        return GRPO_CONFIG["6_9_b"]
    elif param_nums < 12_000_000_000:
        return GRPO_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        return GRPO_CONFIG["12_15_b"]
    elif param_nums < 20_000_000_000:
        return GRPO_CONFIG["15_20_b"]
    elif param_nums < 40_000_000_000:
        return GRPO_CONFIG["20_40_b"]
    elif param_nums < 80_000_000_000:
        return GRPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True,
        }


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "vllm_gpu_memory_utilization",
        "num_generations",
        "disable_fa",
        "disable_action_mask",
        "beta",
        "environment_name",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    start_cmd = "python"
    run_type = config["distributed"]
    # if gpu_nums > 1 and run_type == "ddp":
    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    if run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_grpo_env.py \
    --request_path {request_path} \
    --environment_name {environment_name} \
    --bf16 True \
    --report_to wandb \
    --output_dir /workspace/data/trained_model \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 1 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --num_generations {num_generations} --vllm_mode colocate --vllm_gpu_memory_utilization {vllm_gpu_memory_utilization} \
    --disable_fa {disable_fa} \
    --disable_action_mask {disable_action_mask} \
    --beta {beta} \
    --num_generations {num_generations} \
    --loss_type dr_grpo \
    --num_iterations 1 \
    --do_eval False \
    --vllm_max_model_length 16384"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear"
        )

    if config.get("use_vllm", True):
        template += " --use_vllm True"
    else:
        template += " --use_vllm False"

    template += f" --log_completions False"

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("tensor_parallel", False):
        template = template + f" --vllm_tensor_parallel_size {gpu_nums}"

    if config.get("use_4bit", False):
        template = (
            template
            + " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
        )

    if config.get("initial_max_turn", 2) != 2:
        template = template + f" --initial_max_turn {config.get('initial_max_turn', 2)}"
    if config.get("rollouts_per_stage", 1280) != 1280:
        template = template + f" --rollouts_per_stage {config.get('rollouts_per_stage', 1280)}"
    if config.get("rollout_warmup_rollouts") is not None:
        template = template + f" --rollout_warmup_rollouts {config.get('rollout_warmup_rollouts')}"
    if config.get("mcts_warmup_optimizer_steps") is not None:
        template = template + f" --mcts_warmup_optimizer_steps {config.get('mcts_warmup_optimizer_steps')}"
        
    print(f"template: {template}", flush=True)
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_grpo_config(param_nums)
    if model_name == "Qwen/Qwen2.5-3B-Instruct":
        config = GRPO_CONFIG["2_4_b_qwen"]
    if model_name == "Qwen/Qwen2-7B-Instruct":
        config = GRPO_CONFIG["6_9_b_qwen2"]
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        config = GRPO_CONFIG["6_9_b_qwen2_5"]
    if model_name in ["mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.2"]:
        config = GRPO_CONFIG["6_9_b_mistral_effbs8"]
    print(f"config: {config}")
    run_config = {
        "epoch_num": 2,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger": False,
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "disable_action_mask": disable_action_mask(model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
        "vllm_gpu_memory_utilization": config.get("vllm_gpu_memory_utilization", 0.4),
        "use_vllm": get_use_vllm(model_architecture, model_name),
        "tensor_parallel": config.get("tensor_parallel", False),
        "use_4bit": config.get("use_4bit", False),
        "beta": config.get("beta", 0.01),
        "num_generations": config.get("num_generations", 4),
        "initial_max_turn": config.get("initial_max_turn", 2),
        "rollouts_per_stage": config.get("rollouts_per_stage", 1280),
        "rollout_warmup_rollouts": config.get("rollout_warmup_rollouts"),
        "mcts_warmup_optimizer_steps": config.get("mcts_warmup_optimizer_steps"),
        "environment_name": train_info.get("dataset_type", {}).get("environment_name"),
        "log_completions": config.get("log_completions", True),
    }

    if model_name == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5":
        run_config["use_lora"] = True

    if "starcoder" in model_name.lower():
        run_config["batch_size"] = int(run_config["batch_size"] / 1.5)

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 10
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 75

    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]

    run_config["eval_batch_size"] = 4
    if run_config["batch_size"] <= 4:
        run_config["eval_batch_size"] = 1

    if not config.get("use_vllm", True):
        run_config["use_vllm"] = False

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    return {"train_request": train_request, "run_cmd": run_cmd}
