"""
EXPLANATION MODULE

Provides human-readable explanations of the result.

This is the XAI component of the system.
"""


def explain(attributes, risk):
    """
    Prints a simple explanation of the decision.

    Args:
        attributes (dict): extracted attributes
        risk (str): final classification
    """

    print(f"\n=== EXPLANATION ===")
    print(f"Risk Classification: {risk.upper()}\n")

    print("Detected scene attributes:")

    for key, (value, confidence) in attributes.items():
        print(f"- {key}: {value} (confidence: {confidence:.2f})")

    print("\nInterpretation:")
    print("The risk level was determined based on the combination of these attributes using predefined logical rules.")