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
        # Handle uncertainty: if confidence is too low, ground as 'unknown'
        if conf < threshold:
            fact = f"{key}(unknown)."
            facts.append(fact)
            continue

        conf_int = int(conf * 100)

        # Fact format: attribute(label, confidence(score)).
        fact = f"{key}({value}, confidence({conf_int}))."
        facts.append(fact)

    return facts