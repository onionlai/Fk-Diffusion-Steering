"""
Pose VAE Training Package

This package provides tools for training a VAE on swimming pose keypoints.
"""

from .data import load_all_keypoints, KeypointDataset, extract_keypoints_from_txt
from .preprocess import normalize_points, to_vector, KEYS, SEL
from .model import VAE, vae_loss, vae_score
from .train import train_vae

__all__ = [
    'load_all_keypoints',
    'KeypointDataset',
    'extract_keypoints_from_txt',
    'normalize_points',
    'to_vector',
    'KEYS',
    'SEL',
    'VAE',
    'vae_loss',
    'vae_score',
    'train_vae',
]

