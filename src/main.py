"""
MAIN PIPELINE

This file connects ALL components:

Image → CLIP → Attributes → ASP → Risk → Explanation
"""
import argparse

from src.perception.extractor import extract_attributes
from src.grounding.grounder import to_asp_facts
from src.reasoning.asp_runner import run_asp
from src.explanation.explainer import explain


def main():

    parser = argparse.ArgumentParser(description="Neuro-Symbolic Urban Risk Classifier")

    parser.add_argument(
        "--image",
        type=str,
        # Change required=True to default=...
        default="dataset/v1.0-mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
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

        # STEP 4: EXPLANATION (Prints the reasoning trace)
        print("\n=== RESULTS ===")
        print(f"Image Tested: {args.image}")
        if isinstance(answer_set, list):
            explain(attributes, answer_set)
        else:
            explain(attributes, answer_set)

    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] reasoning failed: {e}")

    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {e}")

if __name__ == "__main__":
    main()