import argparse
from pathlib import Path

import torch

from src.configs import load_config
from src.dataset import build_dataloaders
from src.loss import build_loss_fn
from src.model import CVAE
from src.utils import (
    set_seed,
    get_device,
    save_yaml_copy,
    save_checkpoint,
    save_losses_csv,
    AverageMeter,
)

def train_one_epoch(model, loss_fn, optimizer, loader, device):
    model.train()
    meter = AverageMeter()

    for images, labels, s1, s2, s3, s4 in loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        s1 = s1.to(device).float()
        s2 = s2.to(device).float()
        s3 = s3.to(device).float()
        s4 = s4.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = model(images, s1, s2, s3, s4, labels)
        loss = loss_fn(recon, images, mu, logvar)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), n=images.size(0))

    return meter.avg

@torch.no_grad()
def evaluate(model, loss_fn, loader, device):
    model.eval()
    meter = AverageMeter()

    for images, labels, s1, s2, s3, s4 in loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        s1 = s1.to(device).float()
        s2 = s2.to(device).float()
        s3 = s3.to(device).float()
        s4 = s4.to(device).float()

        recon, mu, logvar = model(images, s1, s2, s3, s4, labels)
        loss = loss_fn(recon, images, mu, logvar)
        meter.update(loss.item(), n=images.size(0))

    return meter.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # output dir
    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_copy(args.config, out_dir / "config.yaml")

    # seed
    if cfg.seed.value is not None:
        set_seed(cfg.seed.value, deterministic=cfg.seed.deterministic)

    device = get_device(cfg.train.device)

    # data
    train_loader, test_loader = build_dataloaders(cfg)

    # model
    model = CVAE(
        latent_dim=cfg.model.latent_dim,
        s_param_dim=cfg.model.s_param_dim,
        num_layers=cfg.model.num_layers,
        image_size=tuple(cfg.data.image_size),
    ).to(device)

    # loss + optim
    loss_fn = build_loss_fn(cfg, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # training
    best_metric = float("inf")
    best_state = None
    patience = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else None
    bad_count = 0

    train_losses = []
    test_losses = []

    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_one_epoch(model, loss_fn, optimizer, train_loader, device)
        train_losses.append(tr)

        if test_loader is not None:
            te = evaluate(model, loss_fn, test_loader, device)
            test_losses.append(te)
            monitor_value = te
        else:
            te = None
            monitor_value = tr

        if te is None:
            print(f"Epoch {epoch:04d} | train_loss={tr:.6f}")
        else:
            print(f"Epoch {epoch:04d} | train_loss={tr:.6f} | test_loss={te:.6f}")

        # early stopping / best save
        if cfg.train.early_stopping.enabled:
            improved = monitor_value < best_metric
            if improved:
                best_metric = monitor_value
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= patience:
                    print(f"Early stopping at epoch {epoch} (best={best_metric:.6f})")
                    break

    # save checkpoints
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.logging.save_best and best_state is not None:
        save_checkpoint(best_state, ckpt_dir / "best_model.pth")
    else:
        save_checkpoint(model.state_dict(), ckpt_dir / "last_model.pth")

    if cfg.logging.save_losses:
        save_losses_csv(
            out_dir / "losses.csv",
            train_losses=train_losses,
            test_losses=test_losses if test_loader is not None else None,
        )

if __name__ == "__main__":
    main()
