# --------------------------------------------
# For each image:
#   - Run YOLOv8 inference
#   - Save image with bounding boxes
#   - Save center points of boxes for SAM prompt
# --------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
import os

# ------------------------------
# Setup: Paths
# ------------------------------
input_folder = "train"              # Folder with input images
output_folder = "output-dataset"            # Folder to save outputs
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Load Pretrained YOLOv8 Model
# ------------------------------
model = YOLO("yolov8n.pt")        

# ------------------------------
# Process Each Image in Folder
# ------------------------------
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"📁 Found {len(image_files)} image(s) in '{input_folder}'")

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

    print(f"   ➤ Detected {len(boxes)} bounding boxes")

    # --- Draw Boxes ---
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, "YOLO Box", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- Save Output Image ---
    base_name = os.path.splitext(img_file)[0]
    out_image_path = os.path.join(output_folder, f"{base_name}_boxed.jpg")
    cv2.imwrite(out_image_path, image)
    print(f"   💾 Saved boxed image: {out_image_path}")

    # --- Extract Center Points ---
    prompt_points = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        prompt_points.append([cx, cy])

    # --- Save Center Points to .npy ---
    out_points_path = os.path.join(output_folder, f"{base_name}_points.npy")
    np.save(out_points_path, np.array(prompt_points))
    print(f"   💾 Saved prompt points: {out_points_path}")
