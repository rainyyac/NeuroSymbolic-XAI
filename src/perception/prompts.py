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
    "a photo of a street with many pedestrians",      # high density
    "a photo of a street with a few pedestrians",     # medium density
    "a photo of an empty street with no pedestrians"  # low density
]

PEDESTRIAN_LABELS = ["high", "medium", "low"]


# --- TRAFFIC DENSITY ---
TRAFFIC_PROMPTS = [
    "a photo of a busy road with heavy traffic",      # high traffic
    "a photo of a road with moderate traffic",        # medium traffic
    "a photo of an empty road with no cars"           # low traffic
]

TRAFFIC_LABELS = ["high", "medium", "low"]


# --- CROSSWALK PRESENCE ---
CROSSWALK_PROMPTS = [
    "a photo of a visible zebra crossing on the road",     # crosswalk present
    "a photo of a road without any crosswalk markings"     # crosswalk absent
]

CROSSWALK_LABELS = ["present", "absent", "uncertain"]

# --- OBSTRUCTION ---
OBSTRUCTION_PROMPTS = [
    "a photo of a road with construction barriers or obstacles",
    "a photo of a clear road without any obstructions"
]
OBSTRUCTION_LABELS = ["yes", "no"]


# --- EMERGENCY VEHICLE ---
EMERGENCY_PROMPTS = [
    "a photo of an ambulance, fire truck, or police car",
    "a photo of normal traffic without emergency vehicles"
]
EMERGENCY_LABELS = ["yes", "no"]


# --- VULNERABLE USERS (children, cyclists) ---
VULNERABLE_PROMPTS = [
    "a photo of children or cyclists near a road",
    "a photo of a street without children or cyclists"
]
VULNERABLE_LABELS = ["yes", "no"]