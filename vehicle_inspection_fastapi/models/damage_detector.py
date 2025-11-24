import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

from utils.validate_view_parts_position import (
    assign_left_right,
    assign_door_instance,
    assign_tyre_instance,
    analyze_car_view_advanced
)


class DamageDetector:
    def __init__(self, model_path="models/damage_detector.pt"):
        self.model = YOLO(model_path)

        # Map raw class → canonical part
        self.damage_part_map = {
            "bonnet_damage": "bonnet",
            "frontbumper_damage": "front_bumper",
            "rearbumper_damage": "rear_bumper",
            "headlight_damage": "headlight",
            "taillight_damage": "taillight",
            "luggage_door_damage": "luggage_door",
            "door_damage": "door",
            "sidemirror_damage": "sidemirror",
            "frontwindscreen_damage": "front_windscreen",
            "rearwindscreen_damage": "rear_windscreen",
            "tyre_damage": "tyre",
            "fender_damage": "fender",
            "roof_damage": "roof"
        }

    def detect(self, image) -> Dict[str, Any]:
        
        # Accept both filepath & ndarray
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if img is None:
            return {"damages": [], "error": "Invalid image"}

        img_h, img_w = img.shape[:2]
        result = self.model(img)[0]

        damages = []
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            # STAGE 1 — Collect detections for view analysis + instance logic
            for cls, box, conf in zip(classes, boxes, confs):
                if conf < 0.25:
                    continue

                x1, y1, x2, y2 = map(int, box)

                det = {
                    "class_name": names[cls],
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "x_center": (x1 + x2) / 2 / img_w,
                    "y_center": (y1 + y2) / 2 / img_h,
                    'width': (x2 - x1)/img_w,
                    'height': (y2 - y1)/img_h
                }

                detections.append(det)
                det["class_name"] = self.damage_part_map.get(det["class_name"])

            # STAGE 2 — Determine view
            detected_view = analyze_car_view_advanced(detections, img_w)
            print(f"Detected View: {detected_view}")

            # STAGE 3 — Assign instances for each damage class
            for det in detections:

                base_part = det["class_name"]
                # Instance-level enhancement
                if base_part in ["headlight", "sidemirror", "taillight"]:
                    inst = assign_left_right(base_part, det["x_center"], detected_view, img_w)

                elif base_part == "door":
                    inst = assign_door_instance(detections, det, img_w, detected_view)

                elif base_part == "tyre":
                    inst = assign_tyre_instance(detections, det, img_w, detected_view)

                else:
                    inst = base_part  # bumper, bonnet, roof, etc.

                damages.append({
                    "damage_type": base_part,
                    "part": inst,
                    "confidence": det["confidence"],
                    "x_center": det["x_center"]
                })

        return {
            "damages": damages,
            "detected_view": detected_view
        }
