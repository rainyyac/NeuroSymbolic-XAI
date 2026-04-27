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

    # 1. Extract Risk
    risk_atom = next((a for a in as_set if a.startswith("final_risk(")), "final_risk(unknown)")
    risk_label = risk_atom.split("(")[1].rstrip(")")

    print(f"\n" + "=" * 40)
    print(f" NEURO-SYMBOLIC REASONING TRACE")
    print(f"=" * 40)
    print(f"FINAL CLASSIFICATION: {risk_label.upper()}")

    # 2. Logic Trace: Reasons for Risk vs. Reasons for Safety
    if risk_label != "safe": #dangerous
        factors = [a.split("(")[1].rstrip(")") for a in as_set if a.startswith("risk_factor(")]
        print("\n[1] IDENTIFIED RISK FACTORS:")
        for f in factors:
            print(f"  → TRIGGERED: {f.replace('_', ' ').title()}")
    else: #safe
        cleared = [a.split("(")[1].rstrip(")") for a in as_set if a.startswith("cleared_factor(")]
        print("\n[1] SAFETY JUSTIFICATION:")
        for c in cleared:
            print(f"  ✓ CLEARED: {c.replace('_', ' ').title()}")
        print("  ✓ RESULT: No logical conditions for elevated risk were met.")

    # 3. Perception Evidence (The "Grounding")
    print("\n[2] SUPPORTING VISUAL EVIDENCE:")
    for attr_name, (label, confidence) in attributes.items():
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