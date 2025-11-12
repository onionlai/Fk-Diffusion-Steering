# Pose VAE Training Package

This package provides tools for training a VAE on swimming pose keypoints extracted from 2D_pelvis.txt files from [SwimXYZ dataset](https://g-fiche.github.io/research-pages/swimxyz/).

## Structure

- `data.py`: Data loading functions (`load_2d_pelvis_txt`, `coco25_to_coco17`, `extract_keypoints_from_txt`, `load_all_keypoints`, `KeypointDataset`)
- `preprocess.py`: Keypoint normalization functions (`normalize_points`, `to_vector`, `KEYS`, `SEL`)
- `model.py`: VAE model definition (`VAE`, `vae_loss`, `vae_score`)
- `train.py`: Training script and main function (`train_vae`, `main`)

## Usage

### Command Line

```bash
python -m pose_vae_training.train \
    --data_dir "dataset/Freestyle" \
    --output pose_vae_model.pt \
    --epochs 30 \
    --batch_size 256 \
    --lr 5e-5 \
    --latent_dim 16 \
    --device cpu \
    --subdirs Side_water_level Side_underwater Side_above_water
```

### Python API

```python
from pose_vae_training import load_all_keypoints, train_vae, VAE

# Load data
X, W = load_all_keypoints(
    data_dir="dataset/Freestyle",
    subdirs_filter=['Side_water_level', 'Side_underwater', 'Side_above_water']
)
# Note: Assumes complete synthetic data (all files have 25 keypoints, all frames have all 12 keypoints)
# Train model
vae = train_vae(X, W, epochs=30, lr=5e-5, device='cpu')
```
