"""Readable pooling components for the Phi-3 ablation study."""

from .config import MainCampaignConfig
from .pooling import LearnedLayerMix, PoolingHead, last_token_pool, masked_mean_pool
from .statistics import exact_sign_flip_pvalue, paired_bootstrap_interval

__all__ = [
    "LearnedLayerMix",
    "MainCampaignConfig",
    "PoolingHead",
    "exact_sign_flip_pvalue",
    "last_token_pool",
    "masked_mean_pool",
    "paired_bootstrap_interval",
]

