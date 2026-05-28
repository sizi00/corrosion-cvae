import csv
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device_str: str):
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")

def save_yaml_copy(src_yaml_path: str, dst_path: Path):
    dst_path.write_text(Path(src_yaml_path).read_text(encoding="utf-8"), encoding="utf-8")

def save_checkpoint(state_dict, path: Path):
    torch.save(state_dict, str(path))

def save_losses_csv(path: Path, train_losses: List[float], test_losses: Optional[List[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if test_losses is None:
            w.writerow(["epoch", "train_loss"])
            for i, tr in enumerate(train_losses, start=1):
                w.writerow([i, tr])
        else:
            w.writerow(["epoch", "train_loss", "test_loss"])
            for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
                w.writerow([i, tr, te])

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)
