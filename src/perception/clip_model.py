"""
CLIP MODEL MODULE

This module is responsible for the "neural perception" part of the system.

CLIP (Contrastive Language–Image Pretraining) is used to extract semantic
information from an image by comparing it to text descriptions.

IMPORTANT:
- CLIP does NOT classify directly into "risk"
- It only measures similarity between an image and text prompts
"""

import torch
import clip
from PIL import Image

_clip_model = None
_preprocess = None
_device = None


def get_model():
    """
    Lazily loads the CLIP model.

    This ensures:
    - model is loaded only once
    - no loading at import time
    - reusable across the whole program
    """

    global _clip_model, _preprocess, _device

    if _clip_model is None:
        print("Loading CLIP model...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _preprocess = clip.load("ViT-B/32", device=_device)

    return _clip_model, _preprocess, _device


def encode_image(image_path):
    """
    Encodes an image into a feature vector using CLIP.

    This is the FIRST step in the pipeline:
    Image → numerical representation (embedding)

    Args:
        image_path (str): path to image file

    Returns:
        image_features (Tensor): high-dimensional representation of the image
    """

    # Load image, apply CLIP preprocessing (resize, normalize, etc.) and disable gradients
    model, preprocess, device = get_model()
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features