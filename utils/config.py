"""
Config loading with OmegaConf.
Supports:
  • YAML file loading
  • Dotlist CLI overrides  (e.g. training.batch_size=32)
  • Nested default merging  (configs/small.yaml -> configs/default.yaml)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(
    config_name: str = "default",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load a YAML config and optionally merge CLI dotlist overrides.

    Args:
        config_name: Base name without .yaml  (e.g. 'default', 'small')
        overrides:   List of dotlist strings   (e.g. ['training.batch_size=8'])

    Returns:
        Merged OmegaConf DictConfig
    """
    yaml_path = _CONFIG_DIR / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    cfg = OmegaConf.load(yaml_path)

    # Handle 'defaults' key for simple inheritance
    # cfg.defaults is a ListConfig, not a plain Python list — must convert
    if "defaults" in cfg:
        defaults_raw = OmegaConf.to_container(cfg.defaults, resolve=True)
        parent_name  = defaults_raw[0] if isinstance(defaults_raw, list) else defaults_raw
        parent_path  = _CONFIG_DIR / f"{parent_name}.yaml"
        if parent_path.exists():
            parent_cfg = OmegaConf.load(parent_path)
            cfg = OmegaConf.merge(parent_cfg, cfg)
        del cfg["defaults"]

    # Merge CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Auto-fill derived values
    hop = OmegaConf.select(cfg, "audio.hop_length", default=None)
    if hop is None:
        sr = OmegaConf.select(cfg, "audio.sample_rate", default=24000)
        tr = OmegaConf.select(cfg, "audio.token_rate",  default=75)
        OmegaConf.update(cfg, "audio.hop_length", sr // tr, merge=True)

    # Ensure output dirs exist
    os.makedirs(str(OmegaConf.select(cfg, "project.log_dir",        default="logs")),        exist_ok=True)
    os.makedirs(str(OmegaConf.select(cfg, "project.checkpoint_dir", default="checkpoints")), exist_ok=True)

    return cfg


def cfg_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
