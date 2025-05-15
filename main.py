import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import sys

# Add `sam_hq` to system path
sys.path.append(str(Path(__file__).parent / "sam_hq"))

# Now import from segment_anything
from segment_anything import SamPredictor, sam_model_registry

# Paths
YOLO_WEIGHTS = "Yolo_Model/yolov5/runs/train/exp/weights/last.pt"
SAM_CHECKPOINT = "sam_hq_vit_h.pth"
IMAGE_PATH = "inputs/test2.jpg"

# Load YOLOv5 model (inference only)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_WEIGHTS)
yolo_model.eval()

# Load SAM-HQ model
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# --- Function: Use YOLOv5 to get wound bounding boxes ---
def detect_with_yolo(image):
    results = yolo_model(image)
    bboxes = results.xyxy[0].cpu().numpy()  # shape: (num_boxes, 6) → [x1, y1, x2, y2, conf, class]
    return bboxes[:, :4]  # only return [x1, y1, x2, y2]

# --- Function: Use SAM-HQ2 to segment based on YOLO box ---
def segment_with_sam(image, box):
    predictor.set_image(image)
    input_box = np.array(box, dtype=np.float32)
    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    return masks[0]

# --- Visualization ---
def visualize(image, mask, box, save_path):
    masked = image.copy()
    masked[~mask] = 0

    # Draw bounding box
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save the result
    result = np.hstack([image, masked])
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved result to: {save_path}")

# --- Main Pipeline ---
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = detect_with_yolo(image_rgb)
    print(f"Detected {len(boxes)} wounds")

    for idx, box in enumerate(boxes):
        mask = segment_with_sam(image_rgb, box)
        save_path = f"outputs/model_segmented_{idx+1}.png"
        visualize(image_rgb.copy(), mask, box, save_path)

if __name__ == "__main__":
    process_image(IMAGE_PATH)
