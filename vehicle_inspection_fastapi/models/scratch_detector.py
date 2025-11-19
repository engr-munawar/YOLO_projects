import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class ScratchDetector:
    def __init__(self, model_path: str = "models/scratch_detector.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect scratches on the car surface"""
        results = self.model(image)
        result = results[0]
        
        scratches = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf > 0.5:
                    scratches.append({
                        'confidence': float(conf),  # Convert to native float
                        'bbox': [float(x) for x in box.tolist()],  # Convert to native floats
                        'length': float(self._calculate_length(box)),  # Convert to native float
                        'severity': self._assess_scratch_severity(conf, box)
                    })
        
        return {
            'scratches': scratches,
            'total_scratches': len(scratches)
        }

    def _calculate_length(self, box: np.ndarray) -> float:
        """Calculate length of scratch and return as native Python float"""
        x1, y1, x2, y2 = box
        length = max(x2 - x1, y2 - y1)  # Use longer dimension as length
        return float(length)  # Convert to native Python float
    
    def _assess_scratch_severity(self, confidence: float, box: np.ndarray) -> str:
        """Assess scratch severity based on confidence and length"""
        length = self._calculate_length(box)
        
        if confidence > 0.8 and length > 100:
            return "deep"
        elif confidence > 0.6 and length > 50:
            return "moderate"
        else:
            return "light"