"""
ATTRIBUTE EXTRACTION MODULE

This is the MOST IMPORTANT bridge in the system.

It converts:
    CLIP numerical outputs → symbolic labels

This step is called "grounding" in neuro-symbolic AI.
"""

import clip
import torch

from src.perception.clip_model import encode_image, model
from src.perception.prompts import *

def score_dimension(image_features, prompts):
    """
    Computes similarity between the image and a set of text prompts.

    This gives us a probability distribution over the prompts.

    Example:
        ["many pedestrians", "few pedestrians", "no pedestrians"]
        → [0.7, 0.2, 0.1]

    Args:
        image_features (Tensor): encoded image
        prompts (list): list of text descriptions

    Returns:
        probs (list): probability for each prompt
    """

    # Convert text prompts into tokens CLIP understands
    tokens = clip.tokenize(prompts).to(image_features.device)

    with torch.no_grad():
        # Encode text prompts into feature vectors
        text_features = model.encode_text(tokens)

        # Compute similarity (dot product) and normalize with softmax
        logits = (image_features @ text_features.T).softmax(dim=-1)

    return logits[0].cpu().numpy()


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

    return {
        "pedestrian_density": (ped_label, ped_conf),
        "traffic_density": (traffic_label, traffic_conf),
        "crosswalk": (crosswalk_label, crosswalk_conf)
    }