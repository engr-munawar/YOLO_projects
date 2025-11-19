import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class DamageDetector:
    def __init__(self, model_path: str = "models/damage_detector.pt"):
        self.model = YOLO(model_path)
        self.damage_classes = ["door_damage", "headlight_damage", "rearbumper_damage", "bonnet_damage", "windscreen_damage", "rearscreen_damage", "taillight_damage", "roof_damage","luggage_door_damage", "fender_damage", "doorscreen_damage", "frontbumper_damage"]
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect various types of damages"""
        results = self.model(image)
        result = results[0]
        
        damages = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            names = result.names
            
            for box, cls, conf in zip(boxes, classes, confidences):
                if conf >= 0.3:
                    damage_type = names[cls]
                    damages.append({
                    'damage_type': names[cls],
                    'confidence': float(conf),  # Convert to native float
                    #'bbox': [float(x) for x in box.tolist()],  # Convert to native floats
                    'severity': self._assess_severity(conf)
                    })
        
        return {
            "damages": damages,
            "total_damages": len(damages)
        }
    
    def _assess_severity(self, confidence: float) -> str:
        """Assess damage severity based on confidence"""
        if confidence > 0.8:
            return "high"
        elif confidence > 0.6:
            return "medium"
        else:
            return "low"