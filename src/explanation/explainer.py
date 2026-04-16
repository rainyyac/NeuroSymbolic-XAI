"""
EXPLANATION MODULE

Provides human-readable explanations of the result.

This is the XAI component of the system.
"""

from src.perception.prompts import PROMPT_MAP

def explain(attributes, answer_set):
    """
    Reconstructs the reasoning trace from the ASP answer set and neural perception.
    """

    as_set = set(answer_set)

    # 1. Extract Risk Level
    risk_atoms = [a for a in as_set if a.startswith("final_risk(")]
    assert len(risk_atoms) == 1, "ERROR: Non-deterministic risk classification!"
    risk_atom = risk_atoms[0]

    risk_label = risk_atom.split("(")[1].rstrip(")")

    print(f"\n" + "=" * 40)
    print(f" NEURO-SYMBOLIC REASONING TRACE")
    print(f"=" * 40)
    print(f"FINAL CLASSIFICATION: {risk_label.upper()}")

    # 2. Extract Triggered Risk Factors (The "Why")
    # This specifically looks for your risk_factor(...) atoms
    factors = [a.split("(")[1].rstrip(")") for a in as_set if a.startswith("risk_factor(")]

    if factors:
        print("\n[1] IDENTIFIED RISK FACTORS:")
        for factor in factors:
            # Clean up names like 'traffic_conflict' to 'Traffic Conflict'
            pretty_factor = factor.replace('_', ' ').title()
            print(f"  → {pretty_factor}")

    # 3. Perception Evidence (The "Grounding")
    print("\n[2] SUPPORTING VISUAL EVIDENCE:")
    for attr_name, (label, confidence) in attributes.items():
        # Check if this attribute was actually part of the reasoning
        if any(f"{attr_name}({label}" in atom for atom in as_set):
            from src.perception.prompts import PROMPT_MAP
            prompt = PROMPT_MAP.get(attr_name, {}).get(label, "N/A")

            print(f"  • {attr_name.replace('_', ' ').title()}: {label}")
            print(f"    └─ CLIP Perception: \"{prompt}\"")
            print(f"    └─ Confidence: {confidence:.2f}")

    # 4. Final Summary
    print("\n[3] INTERPRETATION:")
    if risk_label == "safe":
        print("The scene is orderly. No high-severity risk factors were entailed by the logic.")
    else:
        print(f"The scene was elevated to {risk_label.upper()} due to: {', '.join(factors).replace('_', ' ')}.")