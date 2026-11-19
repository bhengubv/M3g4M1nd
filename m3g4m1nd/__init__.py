"""M3g4M1nd — a small language model, built end to end.

The name is the joke: 63,912,192 parameters, about 1/2700th of GPT-3.

Inspired by and rebuilt from MiniMind (Apache-2.0). See CREDITS.md; that
project is the reason this one exists.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import Turn, pad_trajectories, stop_is_trainable, trajectory_mask, trim_after_first_stop
from .checkpoints import is_resume_checkpoint, load_weights, read_state_dict
from .config import M3Config
from .data import (
    IGNORE,
    PreferenceDataset,
    PretrainDataset,
    PromptDataset,
    SFTDataset,
    read_jsonl,
)
from .export import config_from_state_dict, export_qwen3, qwen3_config, qwen3_moe_config
from .lora import DEFAULT_TARGETS, apply_lora, load_lora, merge_lora, save_lora
from .losses import (
    distillation_loss,
    dpo_loss,
    generalised_advantage_estimate,
    grpo_per_token_loss,
    masked_cross_entropy,
    ppo_losses,
    sequence_logps,
    whiten,
)
from .model import M3ForCausalLM, M3Model, count_parameters
from .outputs import CausalOutput
from .pipelines import StageResult, run_agent, run_distillation, run_dpo, run_grpo
from .rewards import Rubric, deterministic_scorer, ground_truth_hits
from .rl import GRPOUpdater, UpdateStats, ValueHead
from .rollout import HttpRolloutEngine, RolloutResult, TorchRolloutEngine
from .tokenizer import CHAT_TEMPLATE, load_tokenizer, texts_from_jsonl, train_tokenizer
from .tools import CALCULATOR, Tool, ToolRegistry, safe_eval
from .train import (
    StepPlan,
    Trainer,
    TrainerState,
    batches,
    epoch_order,
    init_distributed,
    is_main_process,
    unwrap,
)

__all__ = [
    "__version__",
    # model
    "M3Config", "M3ForCausalLM", "M3Model", "CausalOutput", "count_parameters",
    # tokenizer
    "CHAT_TEMPLATE", "load_tokenizer", "train_tokenizer", "texts_from_jsonl",
    # data
    "IGNORE", "PretrainDataset", "SFTDataset", "PreferenceDataset",
    "PromptDataset", "read_jsonl",
    # training
    "StepPlan", "Trainer", "TrainerState", "batches", "epoch_order",
    "init_distributed", "is_main_process", "unwrap",
    "load_weights", "read_state_dict", "is_resume_checkpoint",
    # stages
    "StageResult", "run_dpo", "run_grpo", "run_agent", "run_distillation",
    # losses
    "distillation_loss", "dpo_loss", "generalised_advantage_estimate",
    "grpo_per_token_loss", "masked_cross_entropy", "ppo_losses",
    "sequence_logps", "whiten",
    # reinforcement learning
    "GRPOUpdater", "UpdateStats", "ValueHead",
    "TorchRolloutEngine", "HttpRolloutEngine", "RolloutResult",
    "Rubric", "deterministic_scorer", "ground_truth_hits",
    # agent
    "Turn", "trajectory_mask", "trim_after_first_stop", "stop_is_trainable",
    "pad_trajectories", "Tool", "ToolRegistry", "CALCULATOR", "safe_eval",
    # adaptation and export
    "DEFAULT_TARGETS", "apply_lora", "load_lora", "save_lora", "merge_lora",
    "export_qwen3", "qwen3_config", "qwen3_moe_config", "config_from_state_dict",
]
