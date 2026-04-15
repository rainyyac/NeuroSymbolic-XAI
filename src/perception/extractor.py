"""
ATTRIBUTE EXTRACTION MODULE

This is the MOST IMPORTANT bridge in the system.

It converts:
    CLIP numerical outputs → symbolic labels

This step is called "grounding" in neuro-symbolic AI.
"""

import clip
import torch

from src.perception.clip_model import encode_image, get_model
from src.perception.prompts import *

def score_dimension(image_features, prompts):
    """
    Computes similarity between the image and a set of text prompts
    using CLIP's proper scoring mechanism.

    This includes:
    - normalization (cosine similarity)
    - temperature scaling (logit_scale)
    - softmax (probability distribution)
    """

    # Tokenize text prompts
    tokens = clip.tokenize(prompts).to(image_features.device)

    with torch.no_grad():
        # Encode text prompts
        model, _, _ = get_model()
        text_features = model.encode_text(tokens)

        # Normalization: Convert embeddings to unit vectors → cosine similarity
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        logits = image_features_norm @ text_features_norm.T

        # Temperature scaling, this sharpens or smooths the distribution
        logit_scale = model.logit_scale.exp()
        logits = logits * logit_scale

        # Softmax
        probs = logits.softmax(dim=-1)

    return probs[0].cpu().numpy()


def get_label(scores, labels):
    """
    Converts probabilities into a discrete label.

    This is where continuous → symbolic happens.

    Args:
        scores (list): probabilities from CLIP
        labels (list): corresponding labels

    Returns:
        label (str): chosen category
        confidence (float): probability of chosen category
    """

    idx = scores.argmax()
    return labels[idx], scores[idx]


def extract_attributes(image_path):
    """
    Full perception pipeline.

    Converts:
        Image → semantic attributes

    Returns:
        dict of attributes with confidence values
    """

    image_features = encode_image(image_path)

    # --- PEDESTRIANS ---
    ped_scores = score_dimension(image_features, PEDESTRIAN_PROMPTS)
    ped_label, ped_conf = get_label(ped_scores, PEDESTRIAN_LABELS)

    # --- TRAFFIC ---
    traffic_scores = score_dimension(image_features, TRAFFIC_PROMPTS)
    traffic_label, traffic_conf = get_label(traffic_scores, TRAFFIC_LABELS)

    # --- CROSSWALK ---
    crosswalk_scores = score_dimension(image_features, CROSSWALK_PROMPTS)
    crosswalk_label, crosswalk_conf = get_label(crosswalk_scores, CROSSWALK_LABELS)

    # --- OBSTRUCTION ---
    obs_scores = score_dimension(image_features, OBSTRUCTION_PROMPTS)
    obs_label, obs_conf = get_label(obs_scores, OBSTRUCTION_LABELS)

    # --- EMERGENCY VEHICLE ---
    em_scores = score_dimension(image_features, EMERGENCY_PROMPTS)
    em_label, em_conf = get_label(em_scores, EMERGENCY_LABELS)

    # --- VULNERABLE USERS ---
    vu_scores = score_dimension(image_features, VULNERABLE_PROMPTS)
    vu_label, vu_conf = get_label(vu_scores, VULNERABLE_LABELS)

    return {
        "pedestrian_density": (ped_label, ped_conf),
        "traffic_density": (traffic_label, traffic_conf),
        "crosswalk": (crosswalk_label, crosswalk_conf),
        "obstruction": (obs_label, obs_conf),
        "emergency_vehicle": (em_label, em_conf),
        "vulnerable_user": (vu_label, vu_conf)
    }