# coco_to_yolo.py
import os
import json
from PIL import Image

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

coco_json_path = 'images/_annotations.coco.json'
img_folder = 'images'
save_dir = 'labels'

os.makedirs(save_dir, exist_ok=True)

with open(coco_json_path) as f:
    data = json.load(f)

id2filename = {img['id']: img['file_name'] for img in data['images']}
id2size = {img['id']: (img['width'], img['height']) for img in data['images']}

for ann in data['annotations']:
    img_id = ann['image_id']
    filename = id2filename[img_id]
    width, height = id2size[img_id]
    bbox = ann['bbox']
    class_id = ann['category_id'] - 1  # 0-index

    label_file = os.path.join(save_dir, filename.replace('.jpg', '.txt'))
    x, y, w, h = convert((width, height), bbox)

    with open(label_file, 'a') as out:
        out.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
