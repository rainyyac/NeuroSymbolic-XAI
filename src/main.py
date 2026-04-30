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


def main():
    BASE_DIR = pathlib.Path(__file__).parent.parent
    DEFAULT_IMAGE = BASE_DIR / "dataset" / "v1.0-mini" / "samples" / "CAM_FRONT" / "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"

    parser = argparse.ArgumentParser(description="Neuro-Symbolic Urban Risk Classifier")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE),
        help="Path to the urban scene image"
    )
    args = parser.parse_args()

    try:
        # STEP 1: PERCEPTION
        attributes = extract_attributes(args.image)

        # STEP 2: GROUNDING
        facts = to_asp_facts(attributes)

        # STEP 3: REASONING
        answer_set = run_asp(facts)
        risk_atom = next((a for a in answer_set if a.startswith("final_risk(")), "final_risk(unknown)")
        risk_label = risk_atom.split("(")[1].rstrip(")")

        # STEP 4: EXPLANATION (Prints the reasoning trace)
        print("\n=== RESULTS ===")
        print(f"Image Tested: {args.image}")
        explain(attributes, answer_set)
        visualize_result(args.image, risk_label, attributes)

    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] reasoning failed: {e}")

    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {e}")

if __name__ == "__main__":
    main()