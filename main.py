import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from hydra import initialize_config_module, compose, initialize_config_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "sam_hq", "sam_hq2",))

config_dir = os.path.join(
    os.path.dirname(__file__), 
    "sam_hq", "sam_hq2", "sam2", "configs"
)
initialize_config_dir(config_dir=config_dir, version_base="1.3")

# 1️⃣ Import SAM-HQ model and predictor from official repo
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# 🔁 Import YOLOv8
from ultralytics import YOLO

# 2️⃣ Load test image
image_path = "inputs/test1.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Ours SAM-HQ2
checkpoint = "sam_hq/sam_hq2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"


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
checkpoint_path = "sam_hq/sam_hq2/checkpoints/sam2.1_hiera_base_plus.pt"
model_type = "vit_h"  # SAM-HQ supports vit_h, vit_l, vit_b

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build SAM-HQ model using the helper
sam = build_sam2(model_cfg, checkpoint)
sam.to(device)

# 5️⃣ Create predictor and set image
predictor = SAM2ImagePredictor(sam)
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
output_path = "outputs_SAM_HQ2/segmented_result1.png"
plt.savefig(output_path)
print(f"✅ Segmentation saved to: {output_path}")
plt.show()
