"""
EXPLANATION MODULE

Provides human-readable explanations of the result.

This is the XAI component of the system.
"""

from src.perception.prompts import PROMPT_MAP_NEURAL,PROMPT_MAP_SYMBOLIC
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def explain(attributes, answer_set):
    """
    Reconstructs the reasoning trace specifically for Mode B (Neuro-Symbolic).
    """
    as_set = set(answer_set)
    risk_atom = next((a for a in as_set if a.startswith("final_risk(")), "final_risk(unknown)")
    risk_label = risk_atom.split("(")[1].rstrip(")")

    print(f"\n{'=' * 40}\n MODE B: NEURO-SYMBOLIC REASONING TRACE\n{'=' * 40}")
    print(f"LOGIC-BASED CLASSIFICATION: {risk_label.upper()}")

    print("\n[1] SUPPORTING VISUAL EVIDENCE (Grounding Atoms):")
    for attr_name, (label, confidence) in attributes.items():
        if any(f"{attr_name}({label}" in atom for atom in as_set):
            # Pick the first prompt from the list for the console output
            prompts = PROMPT_MAP_SYMBOLIC.get(attr_name, {}).get(label, ["N/A"])
            print(f"  • {attr_name.replace('_', ' ').title()}: {label}")
            print(f"    └─ CLIP Perception: \"{prompts[0]}\"")
            print(f"    └─ Confidence: {confidence:.2f}")

def visualize_result(image_path, mode_b_risk, mode_a_risk, attributes, answer_set, neural_results):
    """
    Creates a side-by-side dashboard and pops up a window if save_path is None.
    """
    fig, (ax_img, ax_table) = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Plot the Image
    img = mpimg.imread(image_path)
    ax_img.imshow(img)
    ax_img.set_title(f"SCENE: {image_path.split('/')[-1]}", fontsize=10)
    ax_img.axis('off')

    # 2. Prepare Comparison Table Data
    # Map Neural keys to logic categories for side-by-side view
    categories = [
        "pedestrian_dense", "traffic_conflict", "vulnerable_exposure",
        "emergency_situation", "infrastructure_failure", "environmental_hazard"
    ]

    table_data = [["Category", "Mode A (Neural)", "Mode B (NeSy Logic)"]]

    for cat in categories:
        # Mode A: Extract highest confidence label from the neural pipeline
        # (Neural results in main.py are stored as {category: (label, confidence)})
        a_label, a_conf = neural_results.get(cat, ("N/A", 0))
        mode_a_str = f"RISK: YES ({a_conf:.2f})" if "safe" not in a_label else f"SAFE ({a_conf:.2f})"

        # Mode B: Check if logic rule fired in ASP
        # Logic rules in NeSy often match the category names
        logic_fired = any(f"risk_factor({cat})" in atom for atom in answer_set)
        mode_b_str = "⚠️ RISK DETECTED" if logic_fired else "✅ LOGICALLY SAFE"

        table_data.append([cat.replace('_', ' ').title(), mode_a_str, mode_b_str])

    # 3. Render Table
    ax_table.axis('tight')
    ax_table.axis('off')
    tbl = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.35, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 3.0)

    # Highlight Title with Final Comparisons
    plt.suptitle(
        f"COMPARATIVE AUDIT DASHBOARD\nMode A Prediction: {mode_a_risk.upper()} | Mode B Logic Result: {mode_b_risk.upper()}",
        fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()