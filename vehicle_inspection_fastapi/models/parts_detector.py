import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class CarPartsDetector:
    def __init__(self, model_path: str = "models/parts_detector.pt"):
        self.model = YOLO(model_path)
        self.expected_parts = {
            "front": ["front_bumper", "headlight", "headlight", "bonnet", "front_windscreen", "sidemirror", "sidemirror"],
            "back": ["taillight", "taillight", "luggage_door", "rear_bumper", "rear_windscreen"],
            "left": ["sidemirror", "door", "tyre", "tyre", "headlight", "taillight"],
            "right": ["sidemirror", "door", "tyre", "tyre", "headlight", "taillight"]
        }
    
    def detect(self, image: np.ndarray, view_type: str) -> Dict[str, Any]:
        """Detect car parts and identify missing ones"""
        results = self.model(image)
        result = results[0]
        
        detected_parts = []
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            names = result.names
            
            for cls, conf in zip(classes, confidences):
                if conf > 0.3:  # Confidence threshold
                    part_name = names[cls]
                    detected_parts.append({
                        "part_name": part_name,
                        "confidence": float(conf)
                    })
        
        # Identify missing parts
        expected = self.expected_parts.get(view_type, [])
        detected_names = [part["part_name"] for part in detected_parts]
        missing_parts = [part for part in expected if part not in detected_names]
        
        return {
            "detected_parts": detected_parts,
            "missing_parts": missing_parts,
            "expected_parts": expected
        }
    
