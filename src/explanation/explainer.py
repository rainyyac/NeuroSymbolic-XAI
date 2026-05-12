"""
EXPLANATION MODULE

Provides human-readable explanations of the result.

This is the XAI component of the system.
"""
import pathlib
import os

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
    # 1. Setup Figure and Grid
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#f0f0f0')

    # Header
    plt.suptitle("COMPARATIVE URBAN RISK ANALYSIS", fontsize=24, fontweight='bold', y=0.98)

    # Image Section (Top)
    ax_img = plt.subplot2grid((10, 2), (0, 0), colspan=2, rowspan=3)
    img = mpimg.imread(image_path)
    ax_img.imshow(img)
    ax_img.set_title(f"INPUT IMAGE: {image_path.split('/')[-1]}", fontsize=12, pad=10)
    ax_img.axis('off')

    # Categories to display
    categories = [
        ("safety", "Safety", "final_risk"),
        ("pedestrian_dense", "Pedestrian Dense", "pedestrian_density"),
        ("traffic_conflict", "Traffic Conflict", "traffic_conflict"),
        ("vulnerable_exposure", "Vulnerable Exposure", "vulnerable_exposure"),
        ("emergency_situation", "Emergency Situation", "emergency_situation"),
        ("infrastructure_failure", "Infrastructure Failure", "infrastructure_failure"),
        ("environmental_hazard", "Environmental Hazard", "environmental_hazard")
    ]

    # Mapping NeSy attributes to their Logic Parents (Ontology Trace)
    ontology_map = {
        "child": "pedestrian", "adult_pedestrian": "pedestrian", "cyclist": "vulnerable_actor",
        "ambulance": "emergency_vehicle", "police_car": "emergency_vehicle", "fire_truck": "emergency_vehicle",
        "traffic_cone": "obstruction", "construction_barrier": "obstruction", "road_debris": "obstruction"
    }

    # 2. Draw the comparison rows
    for i, (key, title, asp_atom) in enumerate(categories):
        row_idx = 3 + i  # Start after the image

        # --- LEFT: Neural Pipeline ---
        ax_a = plt.subplot2grid((10, 2), (row_idx, 0))
        ax_a.set_facecolor('#fffafa')

        a_label, a_conf = neural_results.get(key, ("Unknown", 0.0))
        # Format label for Safety vs others
        a_text = f"{title}\n{'-' * 20}\n"
        if key == "safety":
            status = "Safe" if "safe" in a_label else "Unsafe"
        else:
            status = "Risk Detected" if "safe" not in a_label else "Safe"

        a_text += f"{status} ({a_conf * 100:.1f}%)"

        ax_a.text(0.1, 0.5, a_text, va='center', fontsize=11, fontweight='bold')
        ax_a.set_xticks([]);
        ax_a.set_yticks([])
        for spine in ax_a.spines.values(): spine.set_visible(True)

        # --- RIGHT: Neuro-Symbolic Pipeline ---
        ax_b = plt.subplot2grid((10, 2), (row_idx, 1))

        # Logic to find the "Trace"
        logic_fired = any(
            f"risk_factor({asp_atom})" in atom or f"final_risk({asp_atom})" in atom for atom in answer_set)

        # Find which CLIP perception supported this
        detected_items = []
        ontology_str = ""
        if key == "safety":
            b_status = mode_b_risk.capitalize()
        else:
            b_status = "Detected" if logic_fired else "Logically Safe"

        # Search attributes for items related to this category
        for attr_key, (label, conf) in attributes.items():
            # Check if this specific label (e.g., 'child') belongs to this logic category
            if label in ontology_map:
                detected_items.append(f"• {label} ({conf * 100:.1f}%)")
                ontology_str = f"{label} → {ontology_map[label]}"

        b_text = f"{title}\n{'-' * 20}\n"
        b_text += f"{b_status}\n\n"

        if logic_fired and detected_items:
            b_text += "Detected:\n" + "\n".join(detected_items[:2]) + "\n\n"
            b_text += f"Ontology:\n{ontology_str}\n\n"
            b_text += f"ASP Rule Fired:\n{asp_atom} → {b_status.lower()}"
        elif not logic_fired:
            b_text += "No high-confidence entities \ncleared safety constraints."

        ax_b.text(0.05, 0.5, b_text, va='center', fontsize=9, family='monospace')
        ax_b.set_xticks([]);
        ax_b.set_yticks([])
        for spine in ax_b.spines.values(): spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def generate_html_report(image_path, mode_b_risk, mode_a_risk, attributes, answer_set, neural_results):
    """
    Generates a detailed HTML comparison dashboard with corrected image paths and category-specific traces.
    """
    # 1. Setup paths
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    report_path = BASE_DIR / "evaluation_results" / "latest_audit.html"
    os.makedirs(report_path.parent, exist_ok=True)

    # FIX: Calculate relative path for the image. Relative paths bypass browser file:// blocks.
    rel_image_path = os.path.relpath(image_path, report_path.parent)

    as_set = set(answer_set)

    # Trace filtering: Map which parent classes contribute to which high-level category trace
    category_to_parents = {
        "pedestrian_dense": ["pedestrian"],
        "traffic_conflict": ["vehicle", "vulnerable_actor", "pedestrian"],
        "vulnerable_exposure": ["vulnerable_actor"],
        "emergency_situation": ["emergency_vehicle"],
        "infrastructure_failure": ["crosswalk", "pedestrian_infrastructure"],
        "environmental_hazard": ["obstruction"]
    }

    # Ontology trace mapping (Matches definitions in your ontology.lp)
    ontology_trace_map = {
        "child": "pedestrian", "adult_pedestrian": "pedestrian",
        "cyclist": "vulnerable_actor", "ambulance": "emergency_vehicle",
        "police_car": "emergency_vehicle", "fire_truck": "emergency_vehicle",
        "traffic_cone": "obstruction", "construction_barrier": "obstruction",
        "road_debris": "obstruction", "stalled_vehicle": "obstruction",
        "present": "crosswalk", "absent": "crosswalk"
    }

    categories = [
        ("safety", "Safety", "final_risk"),
        ("pedestrian_dense", "Pedestrian Dense", "pedestrian_density"),
        ("traffic_conflict", "Traffic Conflict", "traffic_conflict"),
        ("vulnerable_exposure", "Vulnerable Exposure", "vulnerable_exposure"),
        ("emergency_situation", "Emergency Situation", "emergency_situation"),
        ("infrastructure_failure", "Infrastructure Failure", "infrastructure_failure"),
        ("environmental_hazard", "Environmental Hazard", "environmental_hazard")
    ]

    html_content = f"""
    <html>
    <head>
        <title>Comparative Urban Risk Analysis</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f7f6; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }}
            .image-box {{ text-align: center; margin-bottom: 30px; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 5px; }}
            .image-box img {{ max-width: 100%; height: auto; border-radius: 3px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}

            .grid {{ display: grid; grid-template-columns: 1fr 1.5fr; gap: 0; border: 1px solid #ccc; }}
            .header {{ background: #007bff; color: white; padding: 12px; font-weight: bold; text-align: center; border: 1px solid #0056b3; }}
            .row {{ display: contents; }}
            .cell {{ padding: 15px; border: 1px solid #eee; min-height: 100px; }}
            .cat-title {{ font-weight: bold; color: #333; margin-bottom: 10px; display: block; }}

            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
            .unsafe {{ background: #ffcccc; color: #b30000; }}
            .safe {{ background: #d4edda; color: #155724; }}

            .trace {{ margin-top: 10px; padding: 10px; background: #f9f9f9; border-left: 3px solid #007bff; font-size: 0.82em; }}
            .trace b {{ color: #555; }}
            ul {{ margin: 5px 0; padding-left: 18px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>COMPARATIVE URBAN RISK ANALYSIS</h1>
            <div class="image-box">
                <img src="{rel_image_path}" alt="Street Scene Image">
                <p style="font-size: 0.8em; color: #666;"><b>SCENE:</b> {os.path.basename(image_path)}</p>
            </div>
            <div class="grid">
                <div class="header">NEURAL PIPELINE (MODE A)</div>
                <div class="header">NEURO-SYMBOLIC PIPELINE (MODE B)</div>
    """

    for key, title, asp_atom in categories:
        # MODE A DATA
        a_label, a_conf = neural_results.get(key, ("Unknown", 0.0))
        a_is_unsafe = "safe" not in a_label.lower() and "absent" not in a_label.lower() and a_label != "low"
        a_status = "Unsafe" if a_is_unsafe else "Safe"
        if key == "pedestrian_dense": a_status = "Dense" if a_is_unsafe else "Safe"
        a_class = "unsafe" if a_is_unsafe else "safe"

        # MODE B DATA (Check if risk_factor for this category fired)
        b_triggered = any(f"risk_factor({key})" in atom for atom in as_set) or \
                      (key == "safety" and "final_risk(dangerous)" in as_set)
        b_status = "Unsafe" if b_triggered else "Safe"
        if key == "pedestrian_dense": b_status = "Dense" if b_triggered else "Safe"
        b_class = "unsafe" if b_triggered else "safe"

        # FILTERED TRACE: Only show entities belonging to this category's parent classes
        detected_items, ontology_lines = [], []
        relevant_parents = category_to_parents.get(key, ["pedestrian", "vulnerable_actor", "emergency_vehicle",
                                                         "obstruction"]) if key == "safety" else category_to_parents.get(
            key, [])

        for attr_name, (label, conf) in attributes.items():
            parent = ontology_trace_map.get(label)
            if parent in relevant_parents or label in relevant_parents:
                detected_items.append(f"<li>{label} ({conf * 100:.1f}%)</li>")
                ontology_lines.append(f"<li>{label} &rarr; {parent}</li>")

        html_content += f"""
                <div class="row">
                    <div class="cell">
                        <span class="cat-title">{title}</span>
                        <span class="badge {a_class}">{a_status} ({a_conf * 100:.1f}%)</span>
                    </div>
                    <div class="cell">
                        <span class="cat-title">{title}</span>
                        <span class="badge {b_class}">{b_status}</span>
                        <div class="trace">
                            <b>LOGIC TRACE:</b>
                            {"<ul>" + "".join(detected_items) + "</ul>" if detected_items else "<ul><li>No relevant entities detected</li></ul>"}
                            <b>Ontology:</b>
                            {"<ul>" + "".join(set(ontology_lines)) + "</ul>" if ontology_lines else "<ul><li>N/A</li></ul>"}
                            <b>ASP Rule Fired:</b><br>
                            {asp_atom if key == "safety" else f"risk_factor({key})"} &rarr; {b_status.lower()}
                        </div>
                    </div>
                </div>
        """

    html_content += "</div></div></body></html>"
    with open(report_path, "w") as f:
        f.write(html_content)
    return report_path