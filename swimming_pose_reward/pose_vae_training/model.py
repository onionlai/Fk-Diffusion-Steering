"""
VAE model definition and loss functions.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE model for keypoint reconstruction."""
    def __init__(self, din, dh=256, dz=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(din, dh),
            nn.ReLU(),
            nn.Linear(dh, dh),
            nn.ReLU()
        )
        self.mu = nn.Linear(dh, dz)
        self.lvar = nn.Linear(dh, dz)
        self.dec = nn.Sequential(
            nn.Linear(dz, dh),
            nn.ReLU(),
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, din)
        )

    def forward(self, x):
        h = self.enc(x)
        mu, lvar = self.mu(h), self.lvar(h)
        std = torch.exp(0.5 * lvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        xhat = self.dec(z)
        return xhat, mu, lvar


def vae_loss(x, xhat, mu, lvar, w=None, beta=0.01):
    """VAE loss: reconstruction + KL divergence."""
    recon = (x - xhat) ** 2 # reconstruction loss (MSE)
    recon = recon.mean()
    kl = -0.5 * torch.mean(1 + lvar - mu.pow(2) - lvar.exp()) # KL divergence (regularization)
    total = recon + beta * kl

    if torch.isnan(total) or torch.isinf(total):
        print(f"Warning: NaN/Inf in loss! recon={recon.item():.4f}, kl={kl.item():.4f}")

    return total, recon.detach(), kl.detach()


@torch.no_grad()
def vae_score(vae, X, reduce="elbo", device='cpu'):
    """Compute VAE score (negative ELBO) for keypoints."""
    vae.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    xhat, mu, lvar = vae(X_t)
    rec = torch.mean((X_t - xhat) ** 2, dim=1)
    kl = -0.5 * torch.mean(1 + lvar - mu.pow(2) - lvar.exp(), dim=1)
    elbo = -(rec + 1e-3 * kl)
    if reduce is None:
        return elbo.detach().cpu().numpy(), rec.detach().cpu().numpy()
    return elbo.detach().cpu().numpy()

