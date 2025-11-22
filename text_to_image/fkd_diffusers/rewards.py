import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2

from image_reward_utils import rm_load
from llm_grading import LLMGrader

# Import pose validity reward
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fk_dir = os.path.dirname(os.path.dirname(current_dir))
    swimming_pose_reward_dir = os.path.join(fk_dir, "swimming_pose_reward")
    if swimming_pose_reward_dir not in sys.path:
        sys.path.insert(0, swimming_pose_reward_dir)
    from pose_reward import do_pose_reward
except ImportError as e:
    print(f"Warning: Could not import pose_reward: {e}, PoseValidity reward will not be available.")
    do_pose_reward = None

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "PoseValidity": None,
    "MixPoseValidityAndHumanPreference": None,
}

# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    if reward_name != "LLMGrader":
        print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)

    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)

    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)

    elif reward_name == "MixPoseValidityAndHumanPreference":
        return do_mix_humanpreference_pose_reward(images=images, prompts=prompts)

    elif reward_name == "PoseValidity":
        return do_pose_reward(images=images) + 3.0

    elif reward_name == "PoseValidity-Clip":
        clip_scores = do_clip_score(images=images, prompts=prompts)
        pose_scores = do_pose_reward(images=images)

        # Normalize clip scores (typically in [-1, 1] range, normalize to [0, 1])
        clip_normalized = [(s + 1) / 2 for s in clip_scores]

        # Normalize pose scores (typically negative, normalize to [0, 1])
        pose_min = min(pose_scores) if pose_scores else 0
        pose_max = max(pose_scores) if pose_scores else 0
        if pose_max > pose_min:
            pose_normalized = [(s - pose_min) / (pose_max - pose_min) for s in pose_scores]
        else:
            pose_normalized = [0.5] * len(pose_scores)  # All same value, use middle

        # 50% CLIP + 50% Pose
        mixed = [0.5 * c + 0.5 * p for c, p in zip(clip_normalized, pose_normalized)]
        return mixed
    else:
        raise ValueError(f"Unknown metric: {reward_name}")

# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

def _normalize(scores, mn=None, mx=None):
    if not scores:
        return []
    if mn is None:
        mn = min(scores)
    if mx is None:
        mx = max(scores)
    if mx > mn:
        normalized_s = [(s - mn) / (mx - mn) for s in scores]
        return [max(0.0, min(1.0, v)) for v in normalized_s]
    return [0.5] * len(scores)

def do_mix_humanpreference_pose_reward(*, images, prompts):
    # pose_reward = do_pose_reward(images=images)
    # return [s + 3.0 for s in pose_reward]
    # return do_human_preference_score(images=images, prompts=prompts)
    human_preference_scores = do_human_preference_score(images=images, prompts=prompts)
    print(f"Human preference scores: {human_preference_scores}")
    pose_scores = do_pose_reward(images=images)
    print(f"Pose validity scores: {pose_scores}")

    hp_norm = _normalize(human_preference_scores, 0.20, 0.35)
    print(f"Normalized human preference scores: {hp_norm}")
    # -10 is not the least possible reward but that is what is used right now in case pose estimation fails
    pose_norm = _normalize(pose_scores, -10.0, 0.0)
    print(f"Normalized pose validity scores: {pose_norm}")

    mixed = [0.5 * c + 0.5 * p for c, p in zip(hp_norm, pose_norm)]
    return mixed

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT

    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()
