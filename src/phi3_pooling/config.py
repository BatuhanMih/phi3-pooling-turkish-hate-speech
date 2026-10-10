"""Locked configuration values for the paired P0/P5 campaign."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MainCampaignConfig:
    model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    model_revision: str = "f39ac1d28e925b323eae81227eaba4464caced4e"
    max_length: int = 128
    training_records: int = 9550
    epochs: int = 3
    optimizer_steps: int = 7164
    learning_rate: float = 8.687484140587463e-05
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1951244703273512
    weight_decay: float = 0.004708867394382477
    label_smoothing: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0697368029059364
    lora_target: str = "qkv_proj"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


PAIRED_SEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842, 942)

