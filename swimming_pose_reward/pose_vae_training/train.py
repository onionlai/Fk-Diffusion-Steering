"""
Main training script for pose VAE.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from .data import load_all_keypoints, KeypointDataset
from .model import VAE, vae_loss


def train_vae(X_real, W_real=None, epochs=30, bs=256, lr=5e-5, dz=16, device='cpu', beta=0.01):
    """Train VAE on keypoint data."""
    # beta is the KL divergence weight, put it low because it's anomaly detection model

    din = X_real.shape[1]
    vae = VAE(din=din, dz=dz).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    ds = KeypointDataset(X_real, W_real)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    for ep in range(1, epochs + 1):
        vae.train()
        tot, n = 0.0, 0
        for x, w in dl:
            x = x.to(device)
            w = w.to(device)
            xhat, mu, lvar = vae(x)
            loss, rec, kl = vae_loss(x, xhat, mu, lvar, w=None, beta=beta) # remove weight w because we are not sure if dataset includes weight (the z), this makes training way more stable
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        if ep % 5 == 0: # print average recon and kl for this epoch
            vae.eval()
            with torch.no_grad():
                sample_x = torch.tensor(X_real[:100], dtype=torch.float32).to(device)
                xhat, mu, lvar = vae(sample_x)
                _, recon_avg, kl_avg = vae_loss(sample_x, xhat, mu, lvar, w=None, beta=beta)
            print(f"epoch {ep}: loss={tot/n:.4f} (recon={recon_avg.item():.4f}, kl={kl_avg.item():.4f})")

    return vae


def main():
    parser = argparse.ArgumentParser(description="Train VAE on swimming pose keypoints")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing 2D_pelvis.txt files")
    parser.add_argument("--output", type=str, default="pose_vae_model.pt",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="Latent dimension")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--subdirs", type=str, nargs='+', default=None,
                        help="Subdirectories to include (e.g., 'Side_water_level Side_underwater Side_above_water'). If not specified, uses all subdirectories.")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load data
    print("Loading keypoints from 2D_pelvis.txt files...")
    X, W = load_all_keypoints(
        args.data_dir,
        subdirs_filter=args.subdirs
    )

    # Compute and save standardization parameters (needed for inference)
    # These are computed inside load_all_keypoints, but we need to recompute here
    # to save them with the model
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8

    # Train VAE
    print("Training VAE...")
    vae = train_vae(
        X, W,
        epochs=args.epochs,
        bs=args.batch_size,
        lr=args.lr,
        dz=args.latent_dim,
        device=device
    )

    # Save model with standardization parameters
    print(f"Saving model to {args.output}...")
    torch.save({
        'model_state_dict': vae.state_dict(),
        'input_dim': X.shape[1],
        'latent_dim': args.latent_dim,
        'X_mean': X_mean,  # Standardization mean (shape: (1, 24))
        'X_std': X_std,    # Standardization std (shape: (1, 24))
    }, args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()

