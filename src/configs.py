from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import yaml

@dataclass
class SeedConfig:
    value: Optional[int]
    deterministic: bool

@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int

@dataclass
class TrainConfig:
    device: str
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    early_stopping: EarlyStoppingConfig

@dataclass
class SplitConfig:
    mode: str  # "none" | "random"
    train_ratio: float = 0.8
    seed: int = 42

@dataclass
class DataConfig:
    image_dir: str
    sparam_paths: Dict[str, str]
    label_parser: str  # "raw_last_token" | "aug_third_token"
    round_digits: Optional[int]
    sparam_label_round_digits: Optional[int]
    channel: str  # "red"
    image_size: List[int]
    normalize_mean: List[float]
    normalize_std: List[float]

@dataclass
class ModelConfig:
    latent_dim: int
    num_layers: int
    s_param_dim: int

@dataclass
class LossConfig:
    use_l1: bool
    use_l2: bool
    use_kld: bool
    use_perceptual: bool
    perceptual_layers: List[int]

@dataclass
class LoggingConfig:
    output_dir: str
    save_best: bool
    save_losses: bool

@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    split: SplitConfig
    seed: SeedConfig
    train: TrainConfig
    model: ModelConfig
    loss: LossConfig
    logging: LoggingConfig

def _require(d: dict, key: str):
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]

def load_config(path: str) -> ExperimentConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    name = _require(raw, "name")

    data_raw = _require(raw, "data")
    data = DataConfig(
        image_dir=_require(data_raw, "image_dir"),
        sparam_paths=_require(data_raw, "sparam_paths"),
        label_parser=_require(data_raw, "label_parser"),
        round_digits=data_raw.get("round_digits", None),
        sparam_label_round_digits=data_raw.get("sparam_label_round_digits", None),
        channel=data_raw.get("channel", "red"),
        image_size=data_raw.get("image_size", [256, 256]),
        normalize_mean=data_raw.get("normalize", {}).get("mean", [0.5]),
        normalize_std=data_raw.get("normalize", {}).get("std", [0.5]),
    )

    split_raw = _require(raw, "split")
    split = SplitConfig(
        mode=_require(split_raw, "mode"),
        train_ratio=split_raw.get("train_ratio", 0.8),
        seed=split_raw.get("seed", 42),
    )

    seed_raw = _require(raw, "seed")
    seed = SeedConfig(
        value=seed_raw.get("value", None),
        deterministic=seed_raw.get("deterministic", False),
    )

    train_raw = _require(raw, "train")
    es_raw = train_raw.get("early_stopping", {}) or {}
    early_stopping = EarlyStoppingConfig(
        enabled=es_raw.get("enabled", False),
        patience=es_raw.get("patience", 100),
    )
    train = TrainConfig(
        device=train_raw.get("device", "cuda"),
        batch_size=_require(train_raw, "batch_size"),
        num_workers=_require(train_raw, "num_workers"),
        epochs=_require(train_raw, "epochs"),
        lr=_require(train_raw, "lr"),
        early_stopping=early_stopping,
    )

    model_raw = _require(raw, "model")
    model = ModelConfig(
        latent_dim=_require(model_raw, "latent_dim"),
        num_layers=_require(model_raw, "num_layers"),
        s_param_dim=_require(model_raw, "s_param_dim"),
    )

    loss_raw = _require(raw, "loss")
    loss = LossConfig(
        use_l1=loss_raw.get("use_l1", False),
        use_l2=loss_raw.get("use_l2", True),
        use_kld=loss_raw.get("use_kld", True),
        use_perceptual=loss_raw.get("use_perceptual", True),
        perceptual_layers=loss_raw.get("perceptual_layers", [3, 8, 15]),
    )

    log_raw = _require(raw, "logging")
    logging = LoggingConfig(
        output_dir=_require(log_raw, "output_dir"),
        save_best=log_raw.get("save_best", True),
        save_losses=log_raw.get("save_losses", True),
    )

    cfg = ExperimentConfig(
        name=name,
        data=data,
        split=split,
        seed=seed,
        train=train,
        model=model,
        loss=loss,
        logging=logging,
    )

    _validate(cfg)
    return cfg

def _validate(cfg: ExperimentConfig):
    if cfg.data.label_parser not in ("raw_last_token", "aug_third_token"):
        raise ValueError(f"Unknown label_parser: {cfg.data.label_parser}")
    if cfg.split.mode not in ("none", "random"):
        raise ValueError(f"Unknown split.mode: {cfg.split.mode}")
    if cfg.data.channel != "red":
        raise ValueError("Only channel='red' is supported (matches your original code).")
    if len(cfg.data.image_size) != 2:
        raise ValueError("data.image_size must be [H, W].")
