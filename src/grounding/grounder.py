"""
GROUNDING MODULE

This module converts structured attributes into ASP facts.

Example:
    ("high", 0.87) → pedestrian_density(high).

This is where the system becomes fully symbolic.
"""

def to_asp_facts(attributes, threshold=0.45):
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
        if conf < threshold:
            fact = f"{key}(unknown)."
            facts.append(fact)
            continue

        fact = f"{key}({value}, confidence({conf:.2f}))."
        facts.append(fact)

    return facts