import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class DentDetector:
    def __init__(self, model_path: str = "models/dent_detector.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect dents in the car body"""
        results = self.model(image)
        result = results[0]
        
        dents = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf > 0.5:
                    dents.append({
                    'confidence': float(conf),  # Convert to native float
                    #'bbox': [float(x) for x in box.tolist()],  # Convert to native floats
                    #'area': float(self._calculate_area(box)),  # Convert to native float
                    #'severity': self._assess_dent_severity(conf, box)
                    })
        
        return {
            "dents": dents,
            "total_dents": len(dents)
        }
    
    def _calculate_area(self, box: np.ndarray) -> float:
        """Calculate area of dent bounding box"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def _assess_dent_severity(self, confidence: float, box: np.ndarray) -> str:
        """Assess dent severity based on confidence and size"""
        area = self._calculate_area(box)
        
        if confidence > 0.8 and area > 1000:
            return "major"
        elif confidence > 0.6 and area > 500:
            return "moderate"
        else:
            return "minor"