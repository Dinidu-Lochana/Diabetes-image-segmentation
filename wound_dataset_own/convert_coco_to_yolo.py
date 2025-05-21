import os
import json
from pathlib import Path
from tqdm import tqdm

# Paths for train
COCO_JSON = "images/train/_annotations.coco.json"
OUTPUT_DIR = "labels/train"
IMAGE_DIR = "images/train"

# Paths for valid 
# COCO_JSON = "images/valid/_annotations.coco.json"
# OUTPUT_DIR = "labels/valid"
# IMAGE_DIR = "images/valid"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load COCO data
with open(COCO_JSON) as f:
    data = json.load(f)

# Map image ID to file name
id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# One class: wound
class_id = 0

# For each annotation
for ann in tqdm(data["annotations"]):
    image_id = ann["image_id"]
    bbox = ann["bbox"]  # COCO format: [x_min, y_min, width, height]
    x, y, w, h = bbox

    image_filename = id_to_filename[image_id]
    image_path = os.path.join(IMAGE_DIR, image_filename)

    # Get actual image size using PIL
    from PIL import Image
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Convert to YOLO format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_path = os.path.join(OUTPUT_DIR, label_filename)

    with open(label_path, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
