"""
PROMPT MAPPINGS

This module defines the "semantic dimensions" of the scene.

Each dimension represents a concept we want to extract:
- pedestrian density
- traffic density
- crosswalk presence
...

CLIP compares the image to all prompts in a dimension,
and we pick the best matching one. This replaces traditional object detection.
"""

PROMPT_MAP = {
    "pedestrian_density": dict(zip(["high", "medium", "low"], [
        "a photo of a street with many pedestrians",
        "a photo of a street with a few pedestrians",
        "a photo of an empty street with no pedestrians"
    ])),
    "traffic_density": dict(zip(["high", "medium", "low"], [
        "a photo of a busy road with heavy traffic",
        "a photo of a road with moderate traffic",
        "a photo of an empty road with no cars"
    ])),
    "crosswalk": dict(zip(["present", "absent"], [
        "a photo of a visible zebra crossing on the road",
        "a photo of a road without any crosswalk markings"
    ])),
    "obstruction": dict(zip(["yes", "no"], [
        "a photo of a road with construction barriers or obstacles",
        "a photo of a clear road without any obstructions"
    ])),
    "emergency_vehicle": dict(zip(["yes", "no"], [
        "a photo of an ambulance, fire truck, or police car",
        "a photo of normal traffic without emergency vehicles"
    ])),
    "vulnerable_user": dict(zip(["yes", "no"], [
        "a photo of children or cyclists near a road",
        "a photo of a street without children or cyclists"
    ]))
}