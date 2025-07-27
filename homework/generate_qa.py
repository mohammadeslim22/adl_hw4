import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),   # Green for karts
    2: (255, 0, 0),   # Red for track boundaries
    3: (0, 0, 255),   # Blue for track elements
    4: (255, 255, 0), # Yellow for special elements
    5: (255, 0, 255), # Magenta for special elements
    6: (0, 255, 255), # Cyan for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.
    """
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(
    image_path: str,
    info_path: str,
    thickness: int = 1,
    min_box_size: int = 5,
) -> np.ndarray:
    """
    Draw detection bounding boxes for karts on the image.
    """
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: view_index {view_index} out of range for detections")
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for class_id, track_id, x1, y1, x2, y2 in frame_detections:
        class_id = int(class_id)
        if class_id != 1:
            continue

        x1_s = int(x1 * scale_x)
        y1_s = int(y1 * scale_y)
        x2_s = int(x2 * scale_x)
        y2_s = int(y2 * scale_y)

        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue
        if x2_s < 0 or x1_s > img_width or y2_s < 0 or y1_s > img_height:
            continue

        color = COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([(x1_s, y1_s), (x2_s, y2_s)], outline=color, width=thickness)

    return np.array(pil_image)


def extract_kart_objects(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
) -> list[dict]:
    """
    Extract kart detections, their centers, and mark the ego (center) kart.
    """
    with open(info_path) as f:
        info = json.load(f)
    if view_index >= len(info["detections"]):
        return []
    detections = info["detections"][view_index]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    center_x = img_width / 2
    center_y = img_height / 2

    karts = []
    min_dist = float("inf")
    closest_idx = -1

    for class_id, track_id, x1, y1, x2, y2 in detections:
        class_id = int(class_id)
        if class_id != 1:
            continue

        x1_s = int(x1 * scale_x)
        y1_s = int(y1 * scale_y)
        x2_s = int(x2 * scale_x)
        y2_s = int(y2 * scale_y)

        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue
        if x2_s < 0 or x1_s > img_width or y2_s < 0 or y1_s > img_height:
            continue

        cx = (x1_s + x2_s) / 2
        cy = (y1_s + y2_s) / 2
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5

        karts.append({
            "instance_id": int(track_id),
            "kart_name": f"Kart {track_id}",
            "center": (cx, cy),
            "is_center_kart": False,
        })

        if dist < min_dist:
            min_dist = dist
            closest_idx = len(karts) - 1

    if closest_idx >= 0:
        karts[closest_idx]["is_center_kart"] = True
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Read the track name from the info JSON.
    """
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "Unknown Track")


def generate_qa_pairs(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
) -> list[dict]:
    """
    Create a list of QA dictionaries for a given view.
    """
    qa_pairs = []
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return qa_pairs

    ego = next((k for k in karts if k["is_center_kart"]), None)
    track_name = extract_track_info(info_path)

    # 1. Ego car identification
    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego["kart_name"]})
    # 2. Total number of karts
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})
    # 3. Track identification
    qa_pairs.append({"question": "What track is this?", "answer": track_name})

    cx, cy = ego["center"]
    for other in karts:
        if other["instance_id"] == ego["instance_id"]:
            continue
        ox, oy = other["center"]
        lr = "left" if ox < cx else "right"
        fb = "front" if oy < cy else "back"
        qa_pairs.append({
            "question": f"Is {other['kart_name']} to the left or right of the ego car?",
            "answer": lr
        })
        qa_pairs.append({
            "question": f"Is {other['kart_name']} in front of or behind the ego car?",
            "answer": fb
        })

    # 5. Counting questions
    left_count = sum(1 for k in karts if k["center"][0] < cx)
    right_count = sum(1 for k in karts if k["center"][0] > cx)
    front_count = sum(1 for k in karts if k["center"][1] < cy)
    back_count = sum(1 for k in karts if k["center"][1] > cy)

    qa_pairs.extend([
        {"question": "How many karts are to the left of the ego car?", "answer": str(left_count)},
        {"question": "How many karts are to the right of the ego car?", "answer": str(right_count)},
        {"question": "How many karts are in front of the ego car?", "answer": str(front_count)},
        {"question": "How many karts are behind the ego car?", "answer": str(back_count)}
    ])
    # Filter out empty/zero answers
    qa_pairs = [qa for qa in qa_pairs if qa["answer"] not in (0, "0", "")]

    # Special case for positional questions
    for qa in qa_pairs:
        if "front of or behind" in qa["question"]:
            if qa["answer"] == "behind":
                qa["answer"] = "back"  # Standardize terminology

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Visualize detections and print QA pairs for a single view.
    """
    info_path = Path(info_file)


    base = info_path.stem.replace("_info", "")
    print("base: ", base)
    data_root = info_path.parent.parent.parent

    image_path = data_root / "data" / "train" / f"{base}_{view_index:02d}_im.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    # image_file = list(info_path.parent.glob(f"{base}_{view_index:02d}_im.jpg"))[0]
    image_file = str(image_path)

    annotated = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated)
    plt.axis('off')
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qas = generate_qa_pairs(info_file, view_index)
    print("Question-Answer Pairs:")
    print("-"*50)
    for qa in qas:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-"*50)


def generate_for_split(split: str = "train"):
    """
    Batch-generate QA pairs for all info files in a split.
    """
    data_dir = Path(__file__).parent.parent / "data" / split
    info_files = sorted(data_dir.glob("*_info.json"))
    for info_path in info_files:
        base = info_path.stem.replace("_info", "")
        with open(info_path) as f:
            info = json.load(f)
        all_qas = []
        num_views = len(info.get("detections", []))
        for view_idx in range(num_views):
            qas = generate_qa_pairs(str(info_path), view_idx)
            img_fname = f"{split}/{base}_{view_idx:02d}_im.jpg"
            for qa in qas:
                qa["image_file"] = img_fname
            all_qas.extend(qas)
        out_path = data_dir / f"{base}_qa_pairs.json"
        with open(out_path, "w") as f:
            json.dump(all_qas, f)
        print(f"Wrote {len(all_qas)} QAs → {out_path}")


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_for_split
    })


if __name__ == "__main__":
    main()
