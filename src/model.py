import torch
import torch.nn as nn

class CVAE(nn.Module):
    """
    Keeps your original design:
    - encode: image features + (s11,s21,p11,p21) concat -> mu/logvar
    - decode: z + corrosion_level concat -> image
    - num_layers controls conv depth (same as your develop codes)
    """
    def __init__(self, latent_dim: int, s_param_dim: int, num_layers: int = 3, image_size=(256, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.s_param_dim = s_param_dim
        self.num_layers = num_layers

        h, w = image_size
        if h != w:
            # your original assumes square for init_feature_size logic
            # (can support non-square later if needed)
            raise ValueError("image_size must be square for this implementation (matches your 256x256 setup).")

        self.init_feature_size = h // (2 ** num_layers)

        # encoder
        encoder_layers = []
        in_ch = 1
        out_ch = 32
        for _ in range(num_layers):
            encoder_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_ch = out_ch
            out_ch *= 2
        self.encoder = nn.Sequential(*encoder_layers)

        final_channels = in_ch
        flattened_size = final_channels * self.init_feature_size * self.init_feature_size

        self.fc_mu = nn.Linear(flattened_size + s_param_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size + s_param_dim * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + 1, flattened_size)

        # decoder
        decoder_layers = []
        in_ch = final_channels
        out_ch = in_ch // 2
        for _ in range(num_layers - 1):
            decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            in_ch = out_ch
            out_ch //= 2

        decoder_layers.append(nn.ConvTranspose2d(in_ch, 1, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        self._final_channels = final_channels

    def encode(self, x, s1, s2, s3, s4):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, s1, s2, s3, s4], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, corrosion_level):
        corrosion_level = corrosion_level.view(z.size(0), -1)
        z = torch.cat([z, corrosion_level], dim=1)
        h = self.fc_decode(z)
        h = h.view(-1, self._final_channels, self.init_feature_size, self.init_feature_size)
        return self.decoder(h)

    def forward(self, x, s1, s2, s3, s4, corrosion_level):
        mu, logvar = self.encode(x, s1, s2, s3, s4)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, corrosion_level)
        return recon, mu, logvar

