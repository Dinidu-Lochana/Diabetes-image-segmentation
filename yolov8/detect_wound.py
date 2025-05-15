# --------------------------------------------
# detect.py
# Run YOLOv8 on all images in a folder:
#   - Save image with bounding boxes
#   - Save center points as SAM prompt
# --------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.box_utils import boxes_to_centers  # Optional: helper for center points

# ------------------------------
# Setup: Paths
# ------------------------------
input_folder = "images"                  # Folder containing input images
output_folder = "outputs"        # Folder to store outputs
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Load Pretrained YOLOv8 Model
# ------------------------------
model = YOLO("yolov8n.pt")              # Change to your custom model if needed

# ------------------------------
# Get List of Images
# ------------------------------
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"📁 Found {len(image_files)} image(s) in '{input_folder}'")

# ------------------------------
# Loop Through Images
# ------------------------------
for img_file in image_files:
    # --- Load Image ---
    image_path = os.path.join(input_folder, img_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping unreadable image: {img_file}")
        continue

    print(f"\n🔍 Processing '{img_file}'")

    # --- Run YOLO Inference ---
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    print(f"   ➤ Detected {len(boxes)} bounding box(es)")

    # --- Draw Bounding Boxes ---
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, "YOLO Box", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- Save Image with Boxes ---
    base_name = os.path.splitext(img_file)[0]
    out_image_path = os.path.join(output_folder, f"{base_name}_boxed.jpg")
    cv2.imwrite(out_image_path, image)
    print(f"   💾 Saved boxed image: {out_image_path}")

    # --- Extract Center Points for SAM Prompt ---
    if len(boxes) > 0:
        prompt_points = boxes_to_centers(boxes)
    else:
        prompt_points = np.array([])

    # --- Save Center Points to .npy ---
    out_points_path = os.path.join(output_folder, f"{base_name}_points.npy")
    np.save(out_points_path, prompt_points)
    print(f"   💾 Saved prompt points: {out_points_path}")
