import os
import csv
import argparse
import pathlib
from tqdm import tqdm
from src.perception.extractor import extract_attributes
from src.grounding.grounder import to_asp_facts
from src.reasoning.asp_runner import run_asp
from src.explanation.explainer import visualize_result


def generate_report(output_dir, plots_dir):
    """Internal helper to build the HTML report after the batch finishes."""
    report_path = output_dir / "report.html"
    plots = sorted([f for f in os.listdir(str(plots_dir)) if f.endswith('.png')])

    html_content = f"""
    <html><head><title>Batch Experiment Report</title>
    <style>
        body {{ font-family: sans-serif; background: #f4f4f9; padding: 20px; }}
        .grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; width: 620px; }}
        h1 {{ text-align: center; color: #333; }}
        p {{ font-size: 11px; color: #666; }}
    </style>
    </head><body>
    <h1>Experiment Results: {len(plots)} Scenes</h1>
    <div class="grid">
    """
    for plot in plots:
        html_content += f'<div class="card"><p>File: {plot}</p><img src="plots/{plot}" width="600"></div>'

    html_content += "</div></body></html>"
    with open(report_path, "w") as f:
        f.write(html_content)
    return report_path


def main():
    # 1. SETUP PATHS
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    INPUT_DIR = BASE_DIR / "dataset" / "v1.0-mini" / "samples" / "CAM_FRONT"

    # 2. ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="Run Batch Neuro-Symbolic Experiment")
    parser.add_argument("--start", type=int, default=0, help="Index of the first image to process")
    parser.add_argument("--size", type=int, default=10, help="Number of images to process")
    parser.add_argument("--name", type=str, default="experiment_1", help="Folder name for results")
    args = parser.parse_args()

    # 3. CREATE OUTPUT DIRECTORIES
    experiment_dir = BASE_DIR / "evaluation_results" / args.name
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 4. GET IMAGE LIST
    all_images = sorted([f for f in os.listdir(str(INPUT_DIR)) if f.endswith('.jpg')])
    test_set = all_images[args.start: args.start + args.size]

    if not test_set:
        print(f"❌ No images found for range {args.start} to {args.start + args.size}")
        return

    # 5. EXECUTION LOOP
    csv_path = experiment_dir / "results.csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Index", "Image Name", "Final Risk", "Factors"])

        print(f"🚀 Running experiment '{args.name}' | Start: {args.start} | Size: {args.size}")

        for i, img_name in enumerate(tqdm(test_set)):
            current_idx = args.start + i
            img_path = str(INPUT_DIR / img_name)

            try:
                # Pipeline
                attributes = extract_attributes(img_path)
                facts = to_asp_facts(attributes)
                answer_set = run_asp(facts)

                # Data Extraction
                risk_atom = next((a for a in answer_set if a.startswith("final_risk(")), "final_risk(unknown)")
                risk_label = risk_atom.split("(")[1].rstrip(")")
                factors = "; ".join([a.split("(")[1].rstrip(")") for a in answer_set if a.startswith("risk_factor(")])

                # Save Plot
                out_name = f"{current_idx:03d}_{img_name.replace('.jpg', '.png')}"
                visualize_result(img_path, risk_label, attributes, save_path=str(plots_dir / out_name))

                # Log CSV
                writer.writerow([current_idx, img_name, risk_label, factors])

            except Exception as e:
                print(f"\n[ERROR] Skipping index {current_idx}: {e}")

    # 6. FINALIZE
    report_file = generate_report(experiment_dir, plots_dir)
    print(f"\n✅ Experiment Complete!")
    print(f"📊 CSV: {csv_path}")
    print(f"🌐 Report: {report_file}")
    print(f"👉 To view: open {report_file}")


if __name__ == "__main__":
    main()