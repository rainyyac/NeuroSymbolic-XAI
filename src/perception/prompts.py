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

# Neural Pipline
PROMPT_MAP_NEURAL = {
    "safety": [
        "a safe urban driving scene",
        "normal traffic conditions",
        "a low risk driving environment"
    ],

    "pedestrian_dense": [
        "a crowded pedestrian area",
        "many pedestrians near traffic",
        "a pedestrian dense urban scene"
    ],

    "traffic_conflict": [
        "pedestrians crossing through heavy traffic",
        "a dangerous urban traffic conflict",
        "cars and pedestrians in a risky interaction"
    ],

    "vulnerable_exposure": [
        "a child exposed to traffic danger",
        "a cyclist in a dangerous road situation",
        "a vulnerable road user at risk"
    ],

    "emergency_situation": [
        "an emergency vehicle in dangerous traffic",
        "a dangerous emergency response situation",
        "an ambulance in heavy traffic"
    ],

    "infrastructure_failure": [
        "pedestrians crossing without infrastructure",
        "a dangerous missing crosswalk situation",
        "unsafe pedestrian road conditions"
    ],

    "environmental_hazard": [
        "an obstruction creating traffic danger",
        "dangerous road hazards",
        "an unsafe blocked roadway"
    ]
}

# NEURO-SYMBOLIC PIPELINE
PROMPT_MAP_SYMBOLIC = {
    # PEDESTRIANS
    "pedestrian_type": {

        "child": [
            "a child near a road",
            "a young child in traffic",
            "a child pedestrian"
        ],

        "adult_pedestrian": [
            "an adult pedestrian",
            "a person walking near traffic",
            "pedestrians near the road"
        ],

        "cyclist": [
            "a cyclist riding near vehicles",
            "a bicycle rider in traffic",
            "a cyclist on the street"
        ]
    },

    # EMERGENCY VEHICLES
    "emergency_type": {

        "ambulance": [
            "an ambulance driving on the road",
            "an active ambulance in traffic",
            "an ambulance vehicle"
        ],

        "police_car": [
            "a police car driving in traffic",
            "a police vehicle on the road",
            "an active police car"
        ],

        "fire_truck": [
            "a fire truck in traffic",
            "an emergency fire vehicle",
            "a fire response vehicle"
        ]
    },

    # OBSTRUCTIONS
    "obstruction_type": {

        "traffic_cone": [
            "a traffic cone on the road",
            "construction cones blocking traffic",
            "road cones near vehicles"
        ],

        "construction_barrier": [
            "a construction barrier",
            "roadwork barriers",
            "barriers blocking traffic"
        ],

        "road_debris": [
            "road debris",
            "objects fallen onto the roadway",
            "debris blocking the street"
        ],

        "stalled_vehicle": [
            "a stalled vehicle",
            "a broken down car",
            "a stopped vehicle blocking traffic"
        ]
    },

    # TRAFFIC DENSITY
    "traffic_density_type": {

        "high": [
            "heavy urban traffic",
            "many vehicles on the road",
            "dense traffic congestion"
        ],

        "medium": [
            "moderate traffic",
            "some vehicles on the road",
            "normal city traffic"
        ],

        "low": [
            "light traffic",
            "few vehicles on the road",
            "mostly empty roadway"
        ]
    },

    # CROSSWALK
    "crosswalk_type": {

        "present": [
            "a pedestrian crosswalk",
            "road zebra crossing",
            "painted pedestrian crossing"
        ],

        "absent": [
            "a road without crosswalk markings",
            "missing pedestrian crossing",
            "no zebra crossing visible"
        ]
    }
}