"""
ATTRIBUTE EXTRACTION MODULE

It converts CLIP numerical outputs into symbolic labels (grounding).

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
    results = {}

    for dimension, mapping in PROMPT_MAP.items():
        labels = list(mapping.keys())
        prompts = list(mapping.values())

        scores = score_dimension(image_features, prompts)
        results[dimension] = get_label(scores, labels)

    return results