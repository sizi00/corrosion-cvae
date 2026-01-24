import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from src.configs import ExperimentConfig

def parse_label(filename: str, mode: str, round_digits: Optional[int]) -> float:
    stem = os.path.splitext(os.path.basename(filename))[0]

    if mode == "raw_last_token":
        # original: file.split('_')[-1].replace('.png','')
        val = float(stem.split("_")[-1])
        return val if round_digits is None else round(val, round_digits)

    if mode == "aug_third_token":
        # original with_aug: base_name.split('_')[2]
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"aug_third_token expects at least 3 '_' tokens: {filename}")
        val = float(parts[2])
        return val if round_digits is None else round(val, round_digits)

    raise ValueError(f"Unknown label parser mode: {mode}")

class CorrosionDataset(Dataset):
    """
    Preserves your original logic:
    - images: red channel only -> grayscale
    - label: parsed from filename
    - sparams: 4 tables matched by (last column == label)
    """

    def __init__(
        self,
        image_dir: str,
        sparam_tables: Dict[str, pd.DataFrame],
        label_parser: str,
        round_digits: Optional[int],
        sparam_label_round_digits: Optional[int],
        transform: T.Compose,
    ):
        self.image_dir = image_dir
        self.transform = transform

        self.label_parser = label_parser
        self.round_digits = round_digits

        # copy tables
        self.sparam_tables = {}
        for k, df in sparam_tables.items():
            df2 = df.copy()
            if sparam_label_round_digits is not None:
                df2.iloc[:, -1] = df2.iloc[:, -1].round(sparam_label_round_digits)
            self.sparam_tables[k] = df2

        # load paths + labels
        image_files = []
        labels = {}
        for root, _, files in os.walk(self.image_dir):
            if ".ipynb_checkpoints" in root:
                continue
            for fn in files:
                if fn.endswith(".png"):
                    p = os.path.join(root, fn)
                    image_files.append(p)
                    labels[p] = parse_label(fn, self.label_parser, self.round_digits)

        # match to sparams
        matched = self._match_and_filter(image_files, labels)

        # store filtered lists (bugfix while keeping continue intent)
        self.image_files = [m[0] for m in matched]
        self.labels = {m[0]: m[1] for m in matched}
        self.matched_indices = [m[2] for m in matched]

        print(f"Loaded {len(self.image_files)} images with corresponding S-parameters")

    def _match_and_filter(self, image_files: List[str], labels: Dict[str, float]):
        out = []
        keys = list(self.sparam_tables.keys())

        for img_path in image_files:
            y = labels[img_path]

            rows = []
            for k in keys:
                df = self.sparam_tables[k]
                m = df[df.iloc[:, -1] == y]
                rows.append(m)

            if any(r.empty for r in rows):
                # original with_aug: print and continue
                print(f"ERROR: Corrosion level {y} not found in S-parameter tables for image {img_path}")
                continue

            idxs = [r.index[0] for r in rows]
            if not all(i == idxs[0] for i in idxs):
                raise ValueError(f"Mismatched S-parameter tables for image {img_path}")

            out.append((img_path, y, idxs[0]))

        return out

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        label = self.labels[img_path]

        image = Image.open(img_path).convert("RGB")
        arr = np.asarray(image)
        red = arr[:, :, 0]
        red_img = Image.fromarray(red, "L")

        x = self.transform(red_img)

        matched_idx = self.matched_indices[idx]
        s1 = self.sparam_tables["s11"].iloc[matched_idx, :-1].values
        s2 = self.sparam_tables["s21"].iloc[matched_idx, :-1].values
        s3 = self.sparam_tables["p11"].iloc[matched_idx, :-1].values
        s4 = self.sparam_tables["p21"].iloc[matched_idx, :-1].values

        return (
            x,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(s1, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(s3, dtype=torch.float32),
            torch.tensor(s4, dtype=torch.float32),
        )

def build_dataloaders(cfg: ExperimentConfig):
    # read sparam csvs
    s11 = pd.read_csv(cfg.data.sparam_paths["s11"], index_col="Unnamed: 0")
    s21 = pd.read_csv(cfg.data.sparam_paths["s21"], index_col="Unnamed: 0")
    p11 = pd.read_csv(cfg.data.sparam_paths["p11"], index_col="Unnamed: 0")
    p21 = pd.read_csv(cfg.data.sparam_paths["p21"], index_col="Unnamed: 0")

    transform = T.Compose([
        T.Resize(tuple(cfg.data.image_size)),
        T.ToTensor(),
        T.Normalize(mean=tuple(cfg.data.normalize_mean), std=tuple(cfg.data.normalize_std)),
    ])

    dataset = CorrosionDataset(
        image_dir=cfg.data.image_dir,
        sparam_tables={"s11": s11, "s21": s21, "p11": p11, "p21": p21},
        label_parser=cfg.data.label_parser,
        round_digits=cfg.data.round_digits,
        sparam_label_round_digits=cfg.data.sparam_label_round_digits,
        transform=transform,
    )

    if cfg.split.mode == "none":
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
        )
        return train_loader, None

    # random split
    total = len(dataset)
    train_size = int(cfg.split.train_ratio * total)
    test_size = total - train_size

    gen = torch.Generator().manual_seed(cfg.split.seed)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, test_loader

