from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.configs import ExperimentConfig

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers: List[int], device):
        super().__init__()
        # torchvision version compatibility
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except Exception:
            vgg = models.vgg16(pretrained=True)

        self.vgg = vgg.features[:16].eval().to(device)
        self.layers = set(layers)

        for p in self.vgg.parameters():
            p.requires_grad = False

    def extract(self, x):
        feats = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.layers:
                feats.append(x)
        return feats

    def forward(self, recon_x, x):
        # your original: 1ch -> repeat 3ch
        recon_x = recon_x.repeat(1, 3, 1, 1)
        x = x.repeat(1, 3, 1, 1)

        rf = self.extract(recon_x)
        xf = self.extract(x)

        loss = 0.0
        for a, b in zip(rf, xf):
            loss = loss + F.mse_loss(a, b)
        return loss

def build_loss_fn(cfg: ExperimentConfig, device):
    perceptual = VGGPerceptualLoss(cfg.loss.perceptual_layers, device=device) if cfg.loss.use_perceptual else None

    def loss_fn(recon_x, x, mu, logvar):
        batch = x.size(0)

        total = 0.0

        if cfg.loss.use_l1:
            l1 = F.l1_loss(recon_x, x, reduction="sum") / batch
            total = total + l1

        if cfg.loss.use_l2:
            l2 = F.mse_loss(recon_x, x, reduction="sum") / batch
            total = total + l2

        if cfg.loss.use_kld:
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total = total + kld

        if cfg.loss.use_perceptual and perceptual is not None:
            pl = perceptual(recon_x, x)
            total = total + pl

        return total

    return loss_fn
