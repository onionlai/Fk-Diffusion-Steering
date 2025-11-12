"""
Swimming Pose Validity Reward Function

This module provides functions to compute pose validity rewards using a VAE model.
The reward is based on reconstruction loss: higher reward = better pose validity.
"""

import os
import numpy as np
import torch
from PIL import Image
from typing import List, Union, Optional, Tuple, Dict
from transformers import AutoImageProcessor, VitPoseForPoseEstimation

from pose_vae_training import VAE, normalize_points, to_vector


# Selected keypoints for VAE (same as training)
SELECTED_KEYPOINTS = {
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


class PoseValidityReward:
    """
    Pose validity reward function using VAE reconstruction loss.

    Higher reward = better pose validity (lower reconstruction loss).
    """

    def __init__(
        self,
        vae_model_path: Optional[str] = None,
        device: Optional[str] = None,
        min_confidence: float = 0.0
    ):
        """
        Initialize the pose validity reward function.

        Args:
            vae_model_path: Path to the trained VAE model file (relative to this file's directory)
                           If None, uses default "pose_vae_side_swimming.pt" in the same directory
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            min_confidence: Minimum confidence threshold for keypoint detection
        """
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
              #   device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.min_confidence = min_confidence

        # Set default model path relative to this file's directory
        if vae_model_path is None:
            # Get directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vae_model_path = os.path.join(current_dir, "pose_vae_side_swimming.pt")
        elif not os.path.isabs(vae_model_path):
            # If relative path, make it relative to this file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vae_model_path = os.path.join(current_dir, vae_model_path)

        # Load ViTPose
        print("Loading ViTPose...")
        self.vitpose_processor = AutoImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
        self.vitpose_model.to(self.device)
        self.vitpose_model.eval()
        print("ViTPose loaded")

        # Load VAE model
        print(f"Loading VAE model from {vae_model_path}...")
        if not os.path.exists(vae_model_path):
            raise FileNotFoundError(f"VAE model not found: {vae_model_path}")

        checkpoint = torch.load(vae_model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_dim = checkpoint.get('input_dim', 24)
            latent_dim = checkpoint.get('latent_dim', 16)
            self.vae = VAE(din=input_dim, dz=latent_dim).to(self.device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])

            # Load standardization parameters
            if 'X_mean' in checkpoint and 'X_std' in checkpoint:
                self.X_mean = checkpoint['X_mean']
                self.X_std = checkpoint['X_std']
                print("Standardization parameters loaded from model file")
            else:
                raise ValueError(
                    "Standardization parameters not found in model file. "
                    "Please retrain the model with the updated training script."
                )
        else:
            raise ValueError("Invalid model file format")

        self.vae.eval()
        print("VAE model loaded and ready")

    def extract_keypoints_from_image(
        self,
        image: Union[Image.Image, str],
        min_confidence: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, float]], bool]:
        """
        Extract keypoints from image using ViTPose.

        Args:
            image: PIL Image or path to image file
            min_confidence: Minimum confidence threshold (overrides instance default)

        Returns:
            keypoints_dict: Dict of {name: (x, y)} for selected keypoints
            scores_dict: Dict of {name: score} for confidence scores
            success: bool indicating if extraction was successful
        """
        if min_confidence is None:
            min_confidence = self.min_confidence

        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Invalid image type: {type(image)}")

            width, height = image.size

            # ViTPose requires boxes parameter - use full image box
            boxes = [[[0.0, 0.0, float(width), float(height)]]]

            # Preprocess
            inputs = self.vitpose_processor(image, boxes=boxes, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run ViTPose
            with torch.no_grad():
                outputs = self.vitpose_model(**inputs)

            # Post-process
            pose_results = self.vitpose_processor.post_process_pose_estimation(
                outputs, boxes=boxes
            )[0]

            if len(pose_results) == 0:
                return None, None, False

            person = pose_results[0]
            keypoints = person["keypoints"].cpu().numpy()  # Shape: (17, 2)
            scores = person["scores"].cpu().numpy()  # Shape: (17,)

            # Extract selected keypoints
            keypoints_dict = {}
            scores_dict = {}

            for name, idx in SELECTED_KEYPOINTS.items():
                if idx < len(keypoints):
                    x, y = keypoints[idx, 0], keypoints[idx, 1]
                    score = scores[idx] if idx < len(scores) else 1.0

                    if score >= min_confidence:
                        keypoints_dict[name] = np.array([x, y], dtype=np.float32)
                        scores_dict[name] = float(score)

            # Check if we have enough keypoints
            critical_keys = ['l_hip', 'r_hip', 'l_shoulder', 'r_shoulder']
            has_critical = all(k in keypoints_dict for k in critical_keys)

            if len(keypoints_dict) >= 4 and has_critical:
                return keypoints_dict, scores_dict, True
            else:
                return None, None, False

        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None, None, False

    def normalize_and_vectorize(
        self,
        keypoints_dict: Dict[str, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Normalize keypoints and convert to vector (same as training).

        Args:
            keypoints_dict: Dict of {name: (x, y)} for keypoints

        Returns:
            vector: np.array of shape (24,) - normalized keypoint vector
            success: bool indicating if normalization was successful
        """
        try:
            # Fill missing non-critical keypoints with average of available points
            if len(keypoints_dict) < len(SELECTED_KEYPOINTS):
                avg_pos = np.mean([v for v in keypoints_dict.values()], axis=0)
                for name in SELECTED_KEYPOINTS.keys():
                    if name not in keypoints_dict:
                        keypoints_dict[name] = avg_pos.copy()

            # Normalize (translate to hip center, scale by torso length)
            normalized = normalize_points(keypoints_dict)

            # Convert to vector
            vector = to_vector(normalized)

            return vector, True

        except Exception as e:
            print(f"Error normalizing keypoints: {e}")
            return None, False

    def compute_reward(
        self,
        image: Union[Image.Image, str],
        min_confidence: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float], bool, Optional[Dict[str, np.ndarray]]]:
        """
        Complete pipeline: image -> ViTPose -> normalize -> VAE -> reward.

        Args:
            image: PIL Image or path to image file
            min_confidence: Minimum confidence threshold (overrides instance default)

        Returns:
            reward: float - negative reconstruction loss (higher is better)
            recon_loss: float - reconstruction loss (lower is better)
            success: bool - whether pipeline succeeded
            keypoints_dict: dict - extracted keypoints (for visualization)
        """
        # Step 1: Extract keypoints
        keypoints_dict, scores_dict, success = self.extract_keypoints_from_image(
            image, min_confidence
        )
        if not success:
            return None, None, False, None

        # Step 2: Normalize and vectorize
        vector, success = self.normalize_and_vectorize(keypoints_dict)
        if not success:
            return None, None, False, keypoints_dict

        # Step 3: Apply standardization (same as training)
        vector_std = (vector - self.X_mean.squeeze()) / self.X_std.squeeze()
        vector_std = np.clip(vector_std, -3.0, 3.0)

        # Step 4: Compute VAE reconstruction loss
        with torch.no_grad():
            x_t = torch.tensor(vector_std, dtype=torch.float32).unsqueeze(0).to(self.device)
            xhat, mu, lvar = self.vae(x_t)
            recon_loss = torch.mean((x_t - xhat) ** 2).item()

        # Step 5: Reward = -reconstruction_loss (higher is better)
        reward = -recon_loss

        return reward, recon_loss, True, keypoints_dict

    def score(
        self,
        images: Union[List[Image.Image], List[str], Image.Image, str]
    ) -> List[float]:
        """
        Compute pose validity rewards for images.
        Compatible with rewards.py interface.

        Args:
            images: Single image or list of images (PIL Image or path)
            prompts: Optional prompts (ignored for pose reward)

        Returns:
            List of reward scores (higher is better)
        """
        # Handle single image
        if not isinstance(images, list):
            images = [images]

        rewards = []
        for image in images:
            reward, _, success, _ = self.compute_reward(image)
            if success:
                rewards.append(reward)
            else:
                # If keypoint extraction fails, return a low reward
                rewards.append(-10.0)  # Very low reward for invalid poses

        return rewards


# Global instance (lazy loading)
_pose_reward_instance = None


def do_pose_reward(*, images):
    """
    Compute pose validity rewards for images.
    Compatible with rewards.py interface.

    Args:
        images: List of PIL Images or image paths
        prompts: Optional prompts (ignored for pose reward)

    Returns:
        List of reward scores (higher is better)
    """
    global _pose_reward_instance

    if _pose_reward_instance is None:
        _pose_reward_instance = PoseValidityReward()

    return _pose_reward_instance.score(images)

