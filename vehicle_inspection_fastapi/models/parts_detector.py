import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

from utils.asign_position_view_parts import (
    assign_left_right,
    assign_door_instance,
    assign_tyre_instance,
    analyze_car_view_advanced
)

class CarPartsDetector:
    def __init__(self, model_path: str = "models/parts_detector.pt"):
        self.model = YOLO(model_path)

        self.expected_parts = {
            "front": ["front_bumper", "headlight_left", "headlight_right",
                      "bonnet", "front_windscreen",
                      "sidemirror_left", "sidemirror_right"],

            "back": ["taillight_left", "taillight_right",
                     "luggage_door", "rear_bumper", "rear_windscreen"],

            "left": ["sidemirror_left",
                     "door_front_left", "door_back_left",
                     "tyre_front_left", "tyre_back_left",
                     "headlight_left", "taillight_left"],

            "right": ["sidemirror_right",
                      "door_front_right", "door_back_right",
                      "tyre_front_right", "tyre_back_right",
                      "headlight_right", "taillight_right"]
        }

    def validate_image_view(self, uploaded_view: str, detected_view: str) -> bool:
            """
            Validate if uploaded image matches the expected view.
            supports hybrid views like 'front_left', 'back_right', etc.
            """
            view_mapping = {
                'front': ['front', 'front_left', 'front_right'],
                'back': ['back', 'back_left', 'back_right', 'rear'],
                'side': ['right_side', 'left_side', 'front_left', 'front_right', 'back_left', 'back_right'],
                'right': ['right_side', 'front_right', 'back_right'],
                'left': ['left_side', 'front_left', 'back_left']
            }
            
            uploaded_view = uploaded_view.lower().strip()
            detected_view = detected_view.lower().strip()
            
            for key, allowed_views in view_mapping.items():
                if key in uploaded_view:
                    return detected_view in allowed_views
            
            return False

    def detect(self, image, view_type: str) -> Dict[str, Any]:
    
        # Support both path and ndarray input
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if img is None:
            raise ValueError("Invalid image input")

        img_height, img_width = img.shape[:2]
        results = self.model(img)[0]

        detected_parts = []

        # STAGE 1 — COLLECT ALL DETECTIONS
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            names = results.names

            detections = []
            for cls, box, conf in zip(classes, boxes, confs):
                if conf < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box)

                detection_info = {
                    'class_name': names[cls],
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'x_center': (x1 + x2) / 2 / img_width,
                    'y_center': (y1 + y2) / 2 / img_height,
                    'width': (x2 - x1) / img_width,
                    'height': (y2 - y1) / img_height
                }
                detections.append(detection_info)

            # Stage2: Analyze overall car view from detections
            detected_view = analyze_car_view_advanced(detections, img_width)
            print(f"Detected View: {detected_view}")
            if view_type not in ["front", "back", "left", "right"]:
                view_type = detected_view  # use detected view for optional images
            
            # STAGE 3 — VIEW VALIDATION
            # Validate view for required images
                
            if not self.validate_image_view(view_type, detected_view):
                return {
                    "error": f"Invalid image view",
                    "detected_view": f'{detected_view}',
                    "valid": False
                }

            # STAGE 4 — PROCESS PARTS (ONLY IF VIEW IS VALID)
            for cls, box, conf in zip(classes, boxes, confs):
                if conf < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box)
                raw_part = names[cls]
                conf = float(conf)
                x_center = (x1 + x2) / 2 / img_width

                detection_info = {
                    'class_name': names[cls],
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'x_center': (x1 + x2) / 2 / img_width,
                    'y_center': (y1 + y2) / 2 / img_height,
                    'width': (x2 - x1) / img_width,
                    'height': (y2 - y1) / img_height
                }
                                
                # LEFT / RIGHT LOGIC
                if raw_part in ["headlight", "sidemirror", "taillight"]:
                    inst = assign_left_right(raw_part, box, view_type, img_width)

                elif raw_part == "door":
                    inst = assign_door_instance(detections, detection_info, img_width, view_type)

                elif raw_part == "tyre":
                    inst = assign_tyre_instance(detections, detection_info, img_width, view_type)

                else:
                    inst = raw_part  # bumper/bonnet etc.

                detected_parts.append({
                    "part_name": inst,
                    "confidence": conf,
                    "x_center": x_center
                })

        # STAGE 5 — MISSING PARTS
        expected = self.expected_parts.get(view_type, [])
        detected_names = [p["part_name"] for p in detected_parts]

        missing_parts = [p for p in expected if p not in detected_names]

        return {
            "detected_parts": detected_parts,
            "missing_parts": missing_parts,
            "expected_parts": expected,
            "uploaded_view": view_type,
            "detected_view": detected_view,
            "valid": True
        }

    
