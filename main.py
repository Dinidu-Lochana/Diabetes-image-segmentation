# sam_infer.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# 1️⃣ Import SAM-HQ model and predictor from official repo
from sam_hq.segment_anything.build_sam import build_sam
from sam_hq.segment_anything.predictor import SamPredictor

# 🔁 Import YOLOv8
from ultralytics import YOLO

# 2️⃣ Load test image
image_path = "inputs/test3.png"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 🟡 Load YOLOv8 model
yolo_model = YOLO("runs/detect/train4/weights/best.pt")
results = yolo_model(image_path)[0]  # first result

# 🔁 Convert YOLO bounding boxes to prompt points (center of bbox)
prompt_points = []
for box in results.boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    prompt_points.append([center_x, center_y])

if not prompt_points:
    print("⚠️ No objects detected by YOLO. Exiting.")
    exit()

input_points = np.array(prompt_points)
input_labels = np.ones(len(input_points))  # foreground

# 4️⃣ Load the SAM-HQ model
checkpoint_path = "sam_hq_vit_h.pth"
model_type = "vit_h"  # SAM-HQ supports vit_h, vit_l, vit_b

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build SAM-HQ model using the helper
sam = build_sam(checkpoint=checkpoint_path)
sam.to(device)

# 5️⃣ Create predictor and set image
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# 6️⃣ Predict masks using the prompt points
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False
)

# 7️⃣ Plot results
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for mask in masks:
    plt.imshow(mask, alpha=0.5)  # overlay mask

for pt in prompt_points:
    plt.plot(pt[0], pt[1], 'ro')  # red prompt point

plt.axis('off')
output_path = "outputs/segmented_result3.png"
plt.savefig(output_path)
print(f"✅ Segmentation saved to: {output_path}")
plt.show()
