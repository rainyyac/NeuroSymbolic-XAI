import os
import pathlib


def create_report():
    # 1. Dynamically find the project root
    # __file__ is report.py, .parent is src/, .parent.parent is NeuroSymbolic-XAI/
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    PLOTS_DIR = BASE_DIR / "evaluation_results" / "plots"
    REPORT_PATH = BASE_DIR / "evaluation_results" / "report.html"

    if not PLOTS_DIR.exists():
        print(f"❌ Error: Could not find plots at {PLOTS_DIR}")
        return

    html_content = """
    <html>
    <head>
        <title>Urban Risk Analysis Report</title>
        <style>
            body { font-family: sans-serif; background: #f4f4f9; padding: 20px; }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
            .card { background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; width: 620px; }
            h1 { text-align: center; color: #333; }
            p { font-size: 11px; color: #666; word-break: break-all; margin: 5px 0; }
            img { border-radius: 4px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Batch Analysis Results</h1>
        <div class="grid">
    """

    # Get all plots
    plots = [f for f in os.listdir(str(PLOTS_DIR)) if f.endswith('.png')]
    plots.sort()

    for plot in plots:
        # Relative path from the HTML file to the images
        html_content += f"""
            <div class="card">
                <p><strong>File:</strong> {plot}</p>
                <img src="plots/{plot}" width="600">
            </div>
        """

    html_content += "</div></body></html>"

    with open(REPORT_PATH, "w") as f:
        f.write(html_content)

    print(f"\n✅ SUCCESS!")
    print(f"Location: {REPORT_PATH}")
    print(f"👉 Run this command to open: open {REPORT_PATH}")


if __name__ == "__main__":
    create_report()