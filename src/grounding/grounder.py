"""
GROUNDING MODULE

This module converts structured attributes (statistical probabilities) into ASP facts (logical truths).
It implements confidence-aware grounding and an uncertainty (unknown) state.

Example:
    ("high", 0.87) → pedestrian_density(high).

"""

def to_asp_facts(attributes, threshold=0.45):
    """
    Converts attributes into ASP facts using dual representation.

    - symbolic fact → used by ASP reasoning
    - confidence fact → used for explanation/debugging

    Args:
        attributes (dict): { "attribute_name": (label, confidence_score), ... }
        threshold (float): Minimum confidence required to accept a label.

    Returns:
        list of strings: ASP facts (e.g., "pedestrian_density(high, confidence(0.87)).")
    """

    facts = []

    for key, (value, conf) in attributes.items():
        if conf < threshold: # If confidence is too low -> unknown
            symbolic_value = "unknown"
        else:
            symbolic_value = value

        fact = f"{key}({symbolic_value}."
        facts.append(fact)
        debug = f"{key}_confidence({value}, {conf:.2f})."
        facts.append(debug)

    return facts