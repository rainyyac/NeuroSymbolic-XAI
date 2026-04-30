import os
import csv
import pathlib
from tqdm import tqdm  # pip install tqdm
from src.perception.extractor import extract_attributes
from src.grounding.grounder import to_asp_facts
from src.reasoning.asp_runner import run_asp
from src.explanation.explainer import visualize_result

BASE_DIR = pathlib.Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "v1.0-mini" / "samples" / "CAM_FRONT"
OUTPUT_DIR = BASE_DIR / "evaluation_results"
LIMIT = 50

def run_batch():
    # Create output directories
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    viz_dir = os.path.join(OUTPUT_DIR, "plots")
    pathlib.Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Get list of images
    all_images = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]
    test_set = all_images[:LIMIT]

    # Prepare CSV results
    csv_path = os.path.join(OUTPUT_DIR, "summary_results.csv")

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Name", "Final Risk", "Factors Triggered"])

        print(f"🚀 Starting Batch Analysis of {LIMIT} images...")

        for img_name in tqdm(test_set):
            img_path = str(INPUT_DIR / img_name)

            try:
                # 1. Pipeline execution
                attributes = extract_attributes(img_path)
                facts = to_asp_facts(attributes)
                answer_set = run_asp(facts)

                # 2. Extract results
                risk_atom = next((a for a in answer_set if a.startswith("final_risk(")), "final_risk(unknown)")
                risk_label = risk_atom.split("(")[1].rstrip(")")

                factors = [a.split("(")[1].rstrip(")") for a in answer_set if a.startswith("risk_factor(")]
                factors_str = "; ".join(factors)

                # --- STEP 3: SAVE VISUALIZATION (THE MODIFIED PART) ---
                out_name = f"result_{img_name.replace('.jpg', '.png')}"
                save_to = os.path.join(viz_dir, out_name)

                # This call now saves the file and closes it immediately
                visualize_result(img_path, risk_label, attributes, save_path=save_to)

                # 4. Log to CSV
                writer.writerow([img_name, risk_label, factors_str])

            except Exception as e:
                print(f"\n[ERROR] Skipping {img_name} due to: {e}")

    print(f"\n✅ Batch complete! Look in the '{OUTPUT_DIR}' folder for your data.")


if __name__ == "__main__":
    run_batch()