"""
GROUNDING MODULE

This module converts structured attributes into ASP facts.

Example:
    ("high", 0.87) → pedestrian_density(high).

This is where the system becomes fully symbolic.
"""


def to_asp_facts(attributes):
    """
    Converts attribute dictionary into ASP facts.

    Args:
        attributes (dict):
            {
                "pedestrian_density": ("high", 0.87),
                ...
            }

    Returns:
        list of strings (ASP facts)
    """

    facts = []

    for key, (value, conf) in attributes.items():
        # Create a logical fact
        fact = f"{key}({value})."
        facts.append(fact)

    return facts