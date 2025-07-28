import json
from pathlib import Path
from typing import Dict, Any, List

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    with open(info_path) as f:
        info = json.load(f)

    captions = []

    # 1. Handle ego car - check different possible structures
    ego_kart = None
    if 'karts' in info:
        # Try to find ego kart in different possible formats
        try:
            # Case 1: karts is list of dicts with is_ego field
            ego_kart = next(k for k in info['karts'] if isinstance(k, dict) and k.get('is_ego'))
        except StopIteration:
            try:
                # Case 2: karts is list of dicts with 'ego' field
                ego_kart = next(k for k in info['karts'] if isinstance(k, dict) and k.get('ego'))
            except StopIteration:
                # Case 3: karts is list of strings - use first kart as ego
                if info['karts'] and isinstance(info['karts'][0], str):
                    ego_kart = {'name': info['karts'][0], 'is_ego': True}

    if not ego_kart:
        # Fallback if no ego kart found
        ego_kart = {'name': 'Unknown Kart', 'is_ego': True}

    captions.append(f"{ego_kart.get('name', 'Unknown Kart')} is the ego car.")

    # 2. Counting karts - handle different formats
    num_karts = 0
    if 'karts' in info:
        if isinstance(info['karts'], list):
            num_karts = len(info['karts'])
        elif isinstance(info['karts'], dict):
            num_karts = len(info['karts'].values())
    elif 'detections' in info and view_index < len(info['detections']):
        # Count karts from detections if available
        num_karts = sum(1 for d in info['detections'][view_index] if int(d[0]) == 1)

    captions.append(f"There are {num_karts} karts in the scenario.")

    # 3. Track name - handle missing field
    track_name = info.get('track_name', info.get('track', 'Unknown Track'))
    captions.append(f"The track is {track_name}.")

    # 4. Relative positions - only if we have proper kart data
    if 'karts' in info and isinstance(info['karts'], list) and all(isinstance(k, dict) for k in info['karts']):
        for kart in info['karts']:
            if kart.get('is_ego') or kart.get('ego'):
                continue
            position = kart.get('relative_position', 'near')
            captions.append(f"{kart.get('name', 'Unknown Kart')} is {position} of the ego car.")

    return captions


def generate_caption_pairs(
        info_path: str,
        view_index: int,
        img_width: int = 150,
        img_height: int = 100,
) -> List[Dict[str, str]]:
    """
    Generate (image_file, caption) pairs for a given view.

    Args:
        info_path: Path to the info JSON file
        view_index: Index of the view to generate caption for
        img_width: Width of the image (for coordinate scaling)
        img_height: Height of the image (for coordinate scaling)

    Returns:
        List of dictionaries with 'image_file' and 'caption' keys
    """
    with open(info_path) as f:
        info = json.load(f)

    # Get base filename without _info.json
    base_name = Path(info_path).stem.replace("_info", "")
    image_file = f"{base_name}_{view_index:02d}_im.jpg"

    # Extract kart information
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    if not ego_kart:
        return []

    track_name = extract_track_info(info_path)
    captions = []

    # 1. Basic scene description
    basic_caption = (
        f"In this racing scenario, {ego_kart['kart_name']} is the ego car. "
        f"There are {len(karts)} karts racing on the {track_name} track."
    )
    captions.append({
        "image_file": image_file,
        "caption": basic_caption
    })

    # 2. Positional descriptions
    cx, cy = ego_kart["center"]
    for other in karts:
        if other["instance_id"] == ego_kart["instance_id"]:
            continue

        ox, oy = other["center"]
        lr = "left" if ox < cx else "right"
        fb = "front" if oy < cy else "back"

        position_caption = (
            f"{other['kart_name']} is positioned to the {lr} and {fb} "
            f"of the ego car {ego_kart['kart_name']}."
        )
        captions.append({
            "image_file": image_file,
            "caption": position_caption
        })

    # 3. Counting descriptions
    left_count = sum(1 for k in karts if k["center"][0] < cx)
    right_count = sum(1 for k in karts if k["center"][0] > cx)
    front_count = sum(1 for k in karts if k["center"][1] < cy)
    back_count = sum(1 for k in karts if k["center"][1] > cy)

    counting_caption = (
        f"There are {left_count} karts to the left, {right_count} to the right, "
        f"{front_count} in front, and {back_count} behind the ego car."
    )
    captions.append({
        "image_file": image_file,
        "caption": counting_caption
    })

    return captions


def generate_captions_for_split(
        split: str = "train",
        output_file: str = "train_captions.json"
):
    """
    Batch-generate caption pairs for all info files in a data split.

    Args:
        split: Data split to process ('train', 'valid', etc.)
        output_file: Output JSON file to save captions
    """
    data_dir = Path(__file__).parent.parent / "data" / split
    info_files = sorted(data_dir.glob("*_info.json"))

    all_captions = []

    for info_path in info_files:
        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        for view_idx in range(num_views):
            captions = generate_caption_pairs(str(info_path), view_idx)
            all_captions.extend(captions)

    # Save all captions to output file
    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Generated {len(all_captions)} caption pairs → {output_file}")

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_captions_for_split
    })


if __name__ == "__main__":
    main()
