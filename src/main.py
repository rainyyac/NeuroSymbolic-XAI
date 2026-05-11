"""
MAIN PIPELINE

This file connects ALL components:

Image → CLIP → Attributes → ASP → Risk → Explanation
"""

import argparse
import pathlib
from src.perception.extractor import extract_attributes
from src.grounding.grounder import to_asp_facts
from src.reasoning.asp_runner import run_asp
from src.explanation.explainer import explain, visualize_result
from src.perception.prompts import PROMPT_MAP_NEURAL, PROMPT_MAP_SYMBOLIC


def main():
    BASE_DIR = pathlib.Path(__file__).parent.parent
    DEFAULT_IMAGE = BASE_DIR / "dataset" / "v1.0-mini" / "samples" / "CAM_FRONT" / "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609912404.jpg"

    parser = argparse.ArgumentParser(description="Neuro-Symbolic vs Neural Comparative Audit")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    args = parser.parse_args()

    try:
        print(f"\n🚀 Testing Image: {args.image}\n")

        # --- MODE A: PURE NEURAL (The Black Box) ---
        # Directly extract high-level risk categories from CLIP
        neural_results = extract_attributes(args.image, PROMPT_MAP_NEURAL)

        # Mode A picks the highest scoring category directly
        mode_a_risk = max(neural_results, key=lambda k: neural_results[k][1])
        mode_a_conf = neural_results[mode_a_risk][1]

        # --- MODE B: NEURO-SYMBOLIC (The Rational Actor) ---
        # 1. Perception (Granular attributes)
        attributes_symb = extract_attributes(args.image, PROMPT_MAP_SYMBOLIC)

        # 2. Grounding & Reasoning (Logic Layer)
        facts = to_asp_facts(attributes_symb)
        answer_set = run_asp(facts)

        # 3. Extract Mode B Result
        risk_atom = next((a for a in answer_set if a.startswith("final_risk(")), "final_risk(unknown)")
        mode_b_risk = risk_atom.split("(")[1].rstrip(")")

        # --- RESULTS & COMPARISON ---
        print("=" * 50)
        print(f"MODE A (Neural Only):     {mode_a_risk.upper()} (Conf: {mode_a_conf:.2f})")
        print(f"MODE B (Neuro-Symbolic): {mode_b_risk.upper()}")
        print("=" * 50)

        explain(attributes_symb, answer_set)

        # Updated call with all necessary data for the table
        visualize_result(
            image_path=args.image,
            mode_b_risk=mode_b_risk,
            mode_a_risk=mode_a_risk,
            attributes=attributes_symb,
            answer_set=answer_set,  # New
            neural_results=neural_results  # New
        )

    except Exception as e:
        print(f"\n[ERROR]: {e}")

if __name__ == "__main__":
    main()