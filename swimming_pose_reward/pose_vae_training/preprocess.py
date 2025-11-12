"""
Keypoint preprocessing and normalization functions.
"""

import numpy as np
import math

# Selected keypoints for training
KEYS = {
    'l_shoulder': 5,   # LShoulder in COCO17
    'r_shoulder': 6,   # RShoulder
    'l_elbow': 7,      # LElbow
    'r_elbow': 8,      # RElbow
    'l_wrist': 9,      # LWrist
    'r_wrist': 10,     # RWrist
    'l_hip': 11,       # LHip
    'r_hip': 12,       # RHip
    'l_knee': 13,      # LKnee
    'r_knee': 14,      # RKnee
    'l_ankle': 15,     # LAnkle
    'r_ankle': 16,     # RAnkle
}
SEL = list(KEYS.keys())
K = len(SEL)  # 12 keypoints * 2 coords = 24 dimensions


def normalize_points(pts, scale_method='torso'):
    """
    Normalize keypoints: translate to hip center, scale by reference length, rotate to horizontal.
    """
    # translate to hip center
    hip_center = 0.5 * (pts['l_hip'] + pts['r_hip'])
    P = {k: (v - hip_center) for k, v in pts.items()}

    # scale by torso (better if it's side view) or shoulder width (better if it's front view)
    if scale_method == 'torso':
        shoulder_center = 0.5 * (P['l_shoulder'] + P['r_shoulder'])
        scale_length = np.linalg.norm(shoulder_center)
        if scale_length < 1e-6:
            raise ValueError("Torso length too small, cannot normalize")
    elif scale_method == 'shoulder':
        scale_length = np.linalg.norm(P['l_shoulder'] - P['r_shoulder'])
        if scale_length < 1e-6:
            raise ValueError("Shoulder width too small, cannot normalize")
    else:
        raise ValueError(f"Unknown scale_method: {scale_method}")

    P = {k: v / scale_length for k, v in P.items()}

    # totate so shoulder-line is horizontal
    # v = P['r_shoulder'] - P['l_shoulder']
    # ang = -math.atan2(v[1], v[0])
    # R = np.array([
    #     [math.cos(ang), -math.sin(ang)],
    #     [math.sin(ang), math.cos(ang)]
    # ], dtype=np.float32)
    # P = {k: (R @ v) for k, v in P.items()}

    return P


def to_vector(P):
    """Convert normalized point dict to (2K,) vector in fixed order SEL."""
    vec = []
    for name in SEL:
        vec += [P[name][0], P[name][1]]
    return np.array(vec, dtype=np.float32)

