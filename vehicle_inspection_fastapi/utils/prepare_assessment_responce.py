import numpy as np
from typing import Dict

def prepare_assessment_response(assessment: Dict) -> Dict:
    """
    Convert all NumPy types to native Python types for JSON serialization
    """
    
    def convert_numpy_types(obj):
        """Recursively convert NumPy types to native Python types"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return obj.tolist()
        else:
            return obj
    
    return convert_numpy_types(assessment)