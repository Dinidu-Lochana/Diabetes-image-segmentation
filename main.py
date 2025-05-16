import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from hydra import initialize_config_dir

# Add the SAM-HQ2 directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "sam_hq", "sam_hq2"))

# Initialize Hydra config
config_dir = os.path.join(
    os.path.dirname(__file__), 
    "sam_hq", "sam_hq2", "sam2", "configs"
)
initialize_config_dir(config_dir=config_dir, version_base="1.3")

# Import SAM-HQ2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import YOLOv8
from ultralytics import YOLO

# -------------------------- Load Image --------------------------
image_path = "inputs/test1.jpg"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# -------------------------- Run YOLOv8 --------------------------
yolo_model = YOLO("runs/detect/train4/weights/best.pt")
results = yolo_model(image_path)[0]  # Get first result

# Extract original and expanded boxes
original_boxes = []
# expanded_boxes = []
prompt_points = []

for box in results.boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = box
    original_boxes.append([x1, y1, x2, y2])

    # # Expand the box slightly
    # x1_new = x1 - 0 * (x2 - x1)
    # x2_new = x2 + 0 * (x2 - x1)
    # y1_new = y1 - 0 * (y2 - y1)
    # y2_new = y2 + 0 * (y2 - y1)
    # expanded_boxes.append([x1, y1 , x2 , y2 ])

    # Get center point
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    prompt_points.append([center_x, center_y])

if not original_boxes:
    print("⚠️ No objects detected by YOLO. Exiting.")
    exit()

# -------------------------- Load SAM-HQ2 --------------------------
checkpoint_path = "sam_hq/sam_hq2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = build_sam2(model_cfg, checkpoint_path)
sam_model.to(device)

predictor = SAM2ImagePredictor(sam_model)
predictor.set_image(image_rgb)

# -------------------------- Predict Masks --------------------------
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for i in range(len(original_boxes)):
    input_box = np.array([original_boxes[i]])         # shape (1, 4)
    input_point = np.array([[prompt_points[i]]])      # shape (1, 1, 2)
    input_label = np.array([[1]])                     # shape (1, 1)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box,
        multimask_output=False
    )

    # Plot mask
    for mask in masks:
        plt.imshow(mask, alpha=0.5)

    # Plot original YOLO box
    x1, y1, x2, y2 = original_boxes[i]
    plt.gca().add_patch(plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        edgecolor='lime',
        facecolor='none',
        linewidth=2
    ))

# -------------------------- Save & Show --------------------------
plt.axis('off')
os.makedirs("outputs_SAM_HQ2_NO", exist_ok=True)
output_path = "outputs_SAM_HQ2_NO/segmented2_with_boxes.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"✅ Segmented result saved to: {output_path}")
plt.show()
