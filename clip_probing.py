import os
import torch
import clip
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


DATAROOT = './dataset'
nusc = NuScenes(version='v1.0-mini', dataroot='./dataset/v1.0-mini', verbose=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#helper func
def get_image_from_sample(sample_token, camera_name='CAM_FRONT'):
    """Retrieves a PIL image from a nuScenes sample token."""
    sample = nusc.get('sample', sample_token)
    data_token = sample['data'][camera_name]
    data = nusc.get('sample_data', data_token)
    full_path = os.path.join(nusc.dataroot, data['filename'])
    return Image.open(full_path)

#Experiment
def run_clip_analysis(scene_idx, output_filename):
    """Loops through a scene and saves CLIP's 'thoughts' to a CSV."""
    attributes = [
        "a photo of a pedestrian",
        "a photo of a red traffic light",
        "a photo of green grass",
        "a photo of a clear road",
        "hazardous driving situation"
    ]
    att_inputs = clip.tokenize(attributes).to(device)

    my_scene = nusc.scene[scene_idx]
    current_sample_token = my_scene['first_sample_token']
    results = []
    num_samples = my_scene['nbr_samples']

    print(f"\nProcessing scene: {my_scene['name']} ({num_samples} samples total)")

    with tqdm(total=num_samples) as pbar:
        while current_sample_token != "":
            # Load and Preprocess Image
            image = get_image_from_sample(current_sample_token)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # AI Inference
            with torch.no_grad():
                logits_per_image, _ = model(image_input, att_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Store Data
            row = {"sample_token": current_sample_token}
            for attribute, prob in zip(attributes, probs):
                row[attribute] = prob
            results.append(row)

            # Move to next sample
            current_sample = nusc.get('sample', current_sample_token)
            current_sample_token = current_sample['next']
            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
    return df


if __name__ == "__main__":
    # Run Day Analysis (Scene 0)
    df_day = run_clip_analysis(0, "clip_day_results.csv")

    # Run Night Analysis (Scene 8)
    df_night = run_clip_analysis(8, "clip_night_results.csv")

    # Final Comparison Printout
    print("\n--- LOGIC COMPARISON ---")
    print(f"Avg Hazard (Day): {df_day['hazardous driving situation'].mean():.2%}")
    print(f"Avg Hazard (Night): {df_night['hazardous driving situation'].mean():.2%}")
    print(f"Avg Grass (Night): {df_night['a photo of green grass'].mean():.2%}")