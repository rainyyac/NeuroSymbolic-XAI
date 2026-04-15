"""
PROMPT DEFINITIONS

This module defines the "semantic dimensions" of the scene.

Each dimension represents a concept we want to extract:
- pedestrian density
- traffic density
- crosswalk presence

CLIP compares the image to ALL prompts in a dimension,
and we pick the best matching one.

This replaces traditional object detection.
"""

# --- PEDESTRIAN DENSITY ---

PEDESTRIAN_PROMPTS = [
    "a street with many pedestrians",      # high density
    "a street with a few pedestrians",     # medium density
    "an empty street with no pedestrians"  # low density
]

PEDESTRIAN_LABELS = ["high", "medium", "low"]


# --- TRAFFIC DENSITY ---

TRAFFIC_PROMPTS = [
    "a busy road with heavy traffic",      # high traffic
    "a road with moderate traffic",        # medium traffic
    "an empty road with no cars"           # low traffic
]

TRAFFIC_LABELS = ["high", "medium", "low"]


# --- CROSSWALK PRESENCE ---

CROSSWALK_PROMPTS = [
    "a visible zebra crossing on the road",     # crosswalk present
    "a road without any crosswalk markings"     # crosswalk absent
]

CROSSWALK_LABELS = ["present", "absent"]