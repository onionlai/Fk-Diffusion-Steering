"""
Data loading and preprocessing for pose VAE training.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import math

from .preprocess import normalize_points, to_vector, KEYS

# Mapping from keypoint names to COCO25 indices (for training data)
# COCO25 order: Nose(0), Neck(1), RShoulder(2), RElbow(3), RWrist(4),
#               LShoulder(5), LElbow(6), LWrist(7), MidHip(8), RHip(9),
#               RKnee(10), RAnkle(11), LHip(12), LKnee(13), LAnkle(14),
#               REye(15), LEye(16), REar(17), LEar(18), ...
KEYS_COCO25 = {
    'l_shoulder': 5,   # LShoulder in COCO25
    'r_shoulder': 2,   # RShoulder
    'l_elbow': 6,      # LElbow
    'r_elbow': 3,      # RElbow
    'l_wrist': 7,      # LWrist
    'r_wrist': 4,      # RWrist
    'l_hip': 12,       # LHip
    'r_hip': 9,        # RHip
    'l_knee': 13,      # LKnee
    'r_knee': 10,      # RKnee
    'l_ankle': 14,     # LAnkle
    'r_ankle': 11,     # RAnkle
}


def load_2d_pelvis_txt(filepath, debug=False):
    """
    Load keypoints from 2D_pelvis.txt file.

    Format: First line is header with keypoint names, subsequent lines are frames.
    Each line: x;y;z for each keypoint, separated by semicolons.
    Decimal separator is comma (European format).

    Returns:
        keypoints: np.array of shape (n_frames, n_keypoints, 3)
        format_type: 'COCO25'
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 2:
        if debug:
            print(f"DEBUG load_2d_pelvis_txt: {filepath} - Too few lines ({len(lines)})")
        return None, None

    # Parse header to get keypoint order
    header = lines[0].strip()
    keypoint_names = header.split(';')
    # Filter out empty strings (from trailing semicolon)
    keypoint_names = [name.strip() for name in keypoint_names if name.strip()]

    # Extract keypoint names (format: "Name.x", "Name.y", "Name.z")
    kp_dict = {}
    for i, name in enumerate(keypoint_names):
        if '.' in name:
            base_name = name.split('.')[0]
            coord = name.split('.')[1]
            if base_name not in kp_dict:
                kp_dict[base_name] = {}
            kp_dict[base_name][coord] = i

    # Parse data lines
    data_lines = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        values_str = line.replace(',', '.').split(';')
        # Filter out empty strings (from trailing semicolon)
        values_str = [v.strip() for v in values_str if v.strip()]
        try:
            values = [float(v) for v in values_str]
            data_lines.append(values)
        except ValueError:
            continue

    if len(data_lines) == 0:
        if debug:
            print(f"DEBUG load_2d_pelvis_txt: {filepath} - No valid data lines after parsing")
        return None, None

    n_frames = len(data_lines)
    # expected_keypoints = len(keypoint_names) // 3  # Should be 25 for COCO25

    line_lengths = [len(line) for line in data_lines]
    if len(set(line_lengths)) > 1:
        most_common_len = max(set(line_lengths), key=line_lengths.count)
        data_lines = [line for line in data_lines if len(line) == most_common_len]
        n_frames = len(data_lines)

    if n_frames == 0:
        if debug:
            print(f"DEBUG load_2d_pelvis_txt: {filepath} - No frames after filtering inconsistent line lengths")
        return None, None

    # Determine actual keypoints from data
    actual_values_per_line = len(data_lines[0])
    actual_keypoints = actual_values_per_line // 3

    # Require at least 15 keypoints (we need indices up to 14: LAnkle)
    # Some files may have 18 keypoints (missing the last 7 from COCO25), which is fine
    # if actual_keypoints < 15:
    #     if debug:
    #         print(f"DEBUG load_2d_pelvis_txt: {filepath} - Insufficient keypoints: need>=15 (max index 14), got {actual_keypoints} (header has {len(keypoint_names)} values = {len(keypoint_names)//3} keypoints, data has {actual_values_per_line} values = {actual_keypoints} keypoints)")
    #     return None, None

    # Reshape to (n_frames, n_keypoints, 3)
    keypoints = np.array(data_lines).reshape(n_frames, actual_keypoints, 3)

    # If we have more than 25, just use first 25 (shouldn't happen for COCO format)
    if actual_keypoints > 25:
        keypoints = keypoints[:, :25, :]
    # If we have 18 keypoints, we can still use them (we only need indices up to 14)

    return keypoints, 'COCO25'


def extract_keypoints_from_txt(filepath, debug=False):
    """
    Extract and normalize keypoints from 2D_pelvis.txt file.

    Directly extracts 12 keypoints from COCO25 format.

    Assumes complete data (all 12 keypoints present) since this is synthetic data.

    Args:
        filepath: Path to 2D_pelvis.txt file
        debug: Print debug information

    Returns:
        vectors: np.array of shape (n_valid_frames, 24) - normalized keypoint vectors
        weights: np.array of shape (n_valid_frames,) - visibility weights (not used in training)
    """
    keypoints_coco25, format_type = load_2d_pelvis_txt(filepath, debug=debug)
    if keypoints_coco25 is None:
        if debug:
            print(f"DEBUG extract_keypoints_from_txt: {filepath} - load_2d_pelvis_txt returned None")
        return None, None

    # Extract selected keypoints directly from COCO25
    vectors = []
    weights = []
    frames_processed = 0
    frames_skipped_reasons = {'normalization_failed': 0, 'nan_inf': 0}

    for frame_idx, frame_kp in enumerate(keypoints_coco25):
        frames_processed += 1
        # frame_kp shape: (n_keypoints, 3) - (x, y, z)
        # Note: z value is not used in training (uncertain if it's confidence or depth)
        pts = {}

        # Extract 12 keypoints directly from COCO25 using KEYS_COCO25 mapping
        for name, coco25_idx in KEYS_COCO25.items():
            if coco25_idx < len(frame_kp):
                x, y = frame_kp[coco25_idx, 0], frame_kp[coco25_idx, 1]

                if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                    continue

                pts[name] = np.array([x, y], dtype=np.float32)

        # Check for critical keypoints needed for normalization
        # normalize_points requires: l_hip, r_hip, l_shoulder, r_shoulder
        critical_keys = ['l_hip', 'r_hip', 'l_shoulder', 'r_shoulder']
        if not all(k in pts for k in critical_keys):
            frames_skipped_reasons['normalization_failed'] += 1
            continue

        # Require all 12 keypoints (synthetic data should be complete)
        if len(pts) < len(KEYS_COCO25):
            frames_skipped_reasons['normalization_failed'] += 1
            continue

        # Proceed with normalization
        try:
            Pn = normalize_points(pts)
            vec = to_vector(Pn)

            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                frames_skipped_reasons['nan_inf'] += 1
                continue

            vectors.append(vec)

            weights.append(1.0)
        except (ValueError, ZeroDivisionError, KeyError) as e:
            frames_skipped_reasons['normalization_failed'] += 1
            continue

    if len(vectors) == 0:
        if debug:
            print(f"DEBUG extract_keypoints_from_txt: {filepath} - No valid vectors after processing {frames_processed} frames. Skipped reasons: {frames_skipped_reasons}")
        return None, None

    # Only print debug info if debug=True or if there were skipped frames (warnings)
    if debug or (frames_skipped_reasons['normalization_failed'] > 0 or frames_skipped_reasons['nan_inf'] > 0):
        print(f"DEBUG extract_keypoints_from_txt: {filepath} - Frames processed: {frames_processed}, skipped: {frames_skipped_reasons}")

    return np.stack(vectors), np.array(weights, dtype=np.float32)


def load_all_keypoints(data_dir, subdirs_filter=None):
    """
    Load all keypoints from 2D_pelvis.txt files in data directory.

    Assumes complete synthetic data (all files have 25 keypoints, all frames have all 12 keypoints).

    Args:
        data_dir: Root directory containing 2D_pelvis.txt files (recursive search)
        subdirs_filter: List of subdirectory names to include (e.g., ['Side_water_level', 'Side_underwater']).
                       If None, includes all subdirectories.

    Returns:
        X: np.array of shape (n_samples, 24) - normalized keypoint vectors
        W: np.array of shape (n_samples,) - visibility weights (not used in training)
    """
    # Only load 2D_pelvis.txt files from COCO/ subdirectories
    # Pattern: **/COCO/2D_pelvis.txt (explicitly match only 2D_pelvis.txt in COCO/ folders)
    pattern = os.path.join(data_dir, "**", "COCO", "2D_pelvis.txt")
    txt_files = glob.glob(pattern, recursive=True)

    if len(txt_files) == 0:
        raise ValueError(f"No 2D_pelvis.txt files found in COCO/ subdirectories under {data_dir}")

    if subdirs_filter is not None:
        if isinstance(subdirs_filter, str):
            subdirs_filter = [subdirs_filter]

        filtered_files = []
        for f in txt_files:
            for subdir in subdirs_filter:
                if subdir in f:
                    filtered_files.append(f)
                    break
        txt_files = filtered_files
        print(f"Filtered to {len(txt_files)} files in subdirectories: {subdirs_filter}")

    print(f"Found {len(txt_files)} 2D_pelvis.txt files")

    all_vectors = []
    all_weights = []
    skipped_files = 0
    total_frames_loaded = 0

    for i, txt_file in enumerate(txt_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(txt_files)} files... (loaded {total_frames_loaded} frames from {len(all_vectors)} files)")

        vectors, weights = extract_keypoints_from_txt(
            txt_file,
            debug=False
        )
        if vectors is not None and len(vectors) > 0:
            all_vectors.append(vectors)
            all_weights.append(weights)
            total_frames_loaded += len(vectors)
        else:
            skipped_files += 1

    if len(all_vectors) == 0:
        raise ValueError("No valid keypoint data found!")

    X = np.concatenate(all_vectors, axis=0)
    W = np.concatenate(all_weights, axis=0)

    # Additional standardization: clip extreme values and normalize by std
    # This helps with training stability
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std

    # X = np.clip(X, -3.0, 3.0)

    print(f"\nData loading summary:")
    print(f"  Files processed: {len(txt_files)}")
    print(f"  Files with valid data: {len(all_vectors)}")
    print(f"  Files skipped (no valid data): {skipped_files}")
    print(f"  Total frames loaded: {X.shape[0]}")
    print(f"  Keypoint vector dimension: {X.shape[1]}")
    print(f"  Mean confidence weight: {W.mean():.3f}")
    print(f"  After standardization: X=[{X.min():.3f}, {X.max():.3f}], mean={X.mean():.3f}, std={X.std():.3f}")

    if len(all_vectors) == 0:
        print("\nWARNING: No data loaded!")
        print("  Possible issues: filtering too strict, data format mismatch, or invalid keypoints")

    return X, W


class KeypointDataset(Dataset):
    """Dataset for keypoint vectors."""
    def __init__(self, X, W=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.W = torch.tensor(W, dtype=torch.float32) if W is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        w = self.W[i] if self.W is not None else torch.tensor(1.0)
        return x, w

