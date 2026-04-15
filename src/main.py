"""
MAIN PIPELINE

This file connects ALL components:

Image → CLIP → Attributes → ASP → Risk → Explanation
"""

from src.perception.extractor import extract_attributes
from src.grounding.grounder import to_asp_facts
from src.reasoning.asp_runner import run_asp


def main():
    # Path to test image
    image_path = "test.jpg"

    # --- STEP 1: PERCEPTION ---
    attributes = extract_attributes(image_path)

    # --- STEP 2: GROUNDING ---
    facts = to_asp_facts(attributes)

    # --- STEP 3: REASONING ---
    risk = run_asp(facts)

    # --- OUTPUT ---
    print("\n=== RESULTS ===")
    print("Attributes:", attributes)
    print("ASP Facts:", facts)
    print("Final Risk:", risk)


if __name__ == "__main__":
    main()