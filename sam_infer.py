# sam_infer.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# 1️⃣ Import SAM-HQ model and predictor from official repo
from sam_hq.segment_anything.build_sam import build_sam
from sam_hq.segment_anything.predictor import SamPredictor

# 2️⃣ Load test image
image_path = "images/test.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 3️⃣ Define prompt points (use your YOLO output instead)
# Example: center points of bounding boxes
prompt_points = [[200, 300], [400, 500]]  # (x, y) format
input_points = np.array(prompt_points)
input_labels = np.ones(len(input_points))  # All points labeled as "foreground"

# 4️⃣ Load the SAM-HQ model
checkpoint_path = "sam-hq/sam_hq_vit_h.pth"
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
    plt.imshow(mask, alpha=0.5)  # Overlay mask

# Optionally show prompt points
for pt in prompt_points:
    plt.plot(pt[0], pt[1], 'ro')  # red point

plt.axis('off')
output_path = "outputs/segmented_result.png"
plt.savefig(output_path)
print(f"Segmentation saved to: {output_path}")
plt.show()
