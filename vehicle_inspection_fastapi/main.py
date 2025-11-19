from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import cv2
import numpy as np

from models.parts_detector import CarPartsDetector
from models.plate_detector import LicensePlateDetector
from models.damage_detector import DamageDetector
from models.dent_detector import DentDetector
from models.scratch_detector import ScratchDetector
from utils.car_segmentation import detect_and_segment_user_car

app = FastAPI(title="Vehicle Inspection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
parts_detector = CarPartsDetector()
plate_detector = LicensePlateDetector()
damage_detector = DamageDetector()
dent_detector = DentDetector()
scratch_detector = ScratchDetector()

# Create temp directory
os.makedirs("temp_images", exist_ok=True)

# In-memory storage for assessments (in production, use database)
assessments_db = {}

class VehicleInspection:
    def __init__(self):
        self.car_id = str(uuid.uuid4())
        self.assessment = {
            "car_id": self.car_id,
            "license_plate": "",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "missing_parts": [],
                "damaged_parts": [],
                "dents_count": 0,
                "scratches_count": 0,
                "total_assessment_score": 100  # Start with perfect score
            },

            "images": {}
            
        }
    
    def process_image(self, image_path: str, view_type: str) -> Dict[str, Any]:
        """Process a single car image for complete assessment"""
        try:
            print(f"Processing {view_type} image: {image_path}")
            
            # Read and validate image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not read image at {image_path}", "view_type": view_type}
            
            print(f"Image loaded successfully, shape: {image.shape}")
            
            # Segment the car first
            print("Starting car segmentation...")
            segmented_image = detect_and_segment_user_car(image_path)
            if segmented_image is None:
                return {"error": "Could not detect car in image", "view_type": view_type}
            
            print(f"Segmentation successful, segmented image shape: {segmented_image.shape}")
            
            # Save segmented image
            seg_path = f"temp_images/segmented_{self.car_id}_{view_type}.jpg"
            cv2.imwrite(seg_path, segmented_image)
            
            # Run plate detection
            print("Running plate detector...")
            plate_result = plate_detector.get_detailed_detection(segmented_image)
            combined_text = plate_result.get('combined_text', 'NO_PLATE_DETECTED')
            print(f"Plate detection result: {combined_text}")
            
            # Run other detectors
            parts_assessment = parts_detector.detect(segmented_image, view_type)
            damage_assessment = damage_detector.detect(segmented_image)
            dent_assessment = dent_detector.detect(segmented_image)
            scratch_assessment = scratch_detector.detect(segmented_image)
            
            # Store SIMPLIFIED plate data - only combined_text
            plate_data = {
                'combined_text': combined_text
            }
            
            print(f"Storing simplified plate data: {combined_text}")
            
            # Compile image assessment
            image_assessment = {
                "view_type": view_type,
                "license_plate": plate_data,  # Only contains combined_text
                "parts_detected": parts_assessment.get("detected_parts", []),
                "missing_parts": parts_assessment.get("missing_parts", []),
                "damages": damage_assessment.get("damages", []),
                "dents": dent_assessment.get("dents", []),
                "scratches": scratch_assessment.get("scratches", []),
                "assessment_score": self._calculate_image_score(
                    parts_assessment, damage_assessment, dent_assessment, scratch_assessment
                )
            }
            
            # Update overall summary
            self._update_summary(image_assessment)
            
            print(f"Successfully processed {view_type} image with plate: {combined_text}")
            return image_assessment
        
        except Exception as e:
            print(f"Critical error in process_image for {view_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "view_type": view_type,
                "license_plate": {"combined_text": "ERROR"},
                "parts_detected": [],
                "missing_parts": [],
                "damages": [],
                "dents": [],
                "scratches": [],
                "assessment_score": 0,
                "error": str(e)
            }

    def combine_license_plates(self):
        """Simple license plate combination - use the most confident detection"""
        try:
            best_combined_text = ""
            best_source_view = "none"
            best_confidence = 0
            
            print("Starting simple license plate combination...")
            
            for view_type, image_data in self.assessment["images"].items():
                print(f"Checking {view_type} view for plates...")
                
                if (image_data and 
                    'license_plate' in image_data and 
                    image_data['license_plate']):
                    
                    plate_data = image_data['license_plate']
                    combined_text = plate_data.get('combined_text', '')
                    
                    print(f"  {view_type}: '{combined_text}'")
                    
                    # Skip if no plate detected or empty text
                    if not combined_text or combined_text == "NO_PLATE_DETECTED":
                        print(f"  Skipping {view_type} - no plate detected")
                        continue
                    
                    # For simplified version, use text length and content as confidence proxy
                    # Longer text with alphanumeric characters is more likely to be a valid plate
                    confidence = len(combined_text) * 0.1
                    if any(c.isdigit() for c in combined_text) and any(c.isalpha() for c in combined_text):
                        confidence += 0.5
                    
                    print(f"  Confidence: {confidence:.3f}")
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_combined_text = combined_text
                        best_source_view = view_type
                        print(f"  ✅ New best plate from {view_type}")
            
            if best_combined_text:
                self.assessment["license_plate"] = {
                    'combined_text': best_combined_text,
                    'source_view': best_source_view
                }
                print(f"🏆 Selected best plate: '{best_combined_text}' from {best_source_view}")
            else:
                self.assessment["license_plate"] = {
                    'combined_text': '',
                    'source_view': 'none'
                }
                print("❌ No valid plates found in any view")
                
        except Exception as e:
            print(f"❌ Error in simple plate combination: {str(e)}")
            import traceback
            traceback.print_exc()
            self.assessment["license_plate"] = {
                'combined_text': '',
                'source_view': 'error'
            }
    
    def _calculate_image_score(self, parts: Dict, damage: Dict, dent: Dict, scratch: Dict) -> float:
        """Calculate assessment score for an image (0-100)"""
        score = 100
        
        # Deduct for missing parts
        score -= len(parts["missing_parts"]) * 5
        
        # Deduct for damages
        score -= len(damage["damages"]) * 10
        
        # Deduct for dents
        score -= len(dent["dents"]) * 8
        
        # Deduct for scratches
        score -= len(scratch["scratches"]) * 3
        
        return max(0, score)
    
    def _update_summary(self, image_assessment: Dict):
        """Update the overall assessment summary"""
        summary = self.assessment["summary"]
        
        # Track ALL detected parts across ALL images
        all_detected_parts = set()
        
        # First, collect all detected parts from all images
        for view_data in self.assessment["images"].values():
            if "parts_detected" in view_data:
                for part in view_data["parts_detected"]:
                    part_name = part.get("part_name")
                    if part_name:
                        all_detected_parts.add(part_name)
        
        # Define the expected parts for a complete car
        expected_parts = {
            "front_bumper", "headlight", "headlight",  # 2 headlights
            "bonnet", "front_windscreen", "sidemirror", "sidemirror",  # 2 sidemirrors
            "door", "door", "door", "door",  # 4 doors
            "tyre", "tyre", "tyre", "tyre",  # 4 tyres
            "rear_bumper", "taillight", "taillight",  # 2 taillights
            "luggage_door", "rear_windscreen"
        }
        
        # Calculate missing parts: expected parts that were NOT detected in ANY image
        detected_part_counts = {}
        for part_name in all_detected_parts:
            detected_part_counts[part_name] = detected_part_counts.get(part_name, 0) + 1
        
        # For parts that have multiple instances, check if we have the expected count
        missing_parts = []
        for part in expected_parts:
            expected_count = list(expected_parts).count(part)  # Count how many times this part is expected
            detected_count = detected_part_counts.get(part, 0)
            
            if detected_count < expected_count:
                missing_parts.append(part)
        
        # Remove duplicates and sort
        summary["missing_parts"] = sorted(list(set(missing_parts)))
        
        # Update damaged parts with proper mapping
        part_mapping = {
            'bonnet_damage': 'bonnet',
            'frontbumper_damage': 'front_bumper', 
            'headlight_damage': 'headlight',
            'luggage_door_damage': 'luggage_door',
            'rearbumper_damage': 'rear_bumper',
            'taillight_damage': 'taillight',
            'door_damage': 'door',
            'sidemirror_damage': 'sidemirror',
            'frontwindscreen_damage': 'front_windscreen',
            'fender_damage': 'fender',
            'doorscreen_damage': 'door_screen',
            'roof_damage': 'roof'
        }
        
        for damage in image_assessment["damages"]:
            damage_type = damage.get("damage_type", "Unknown")
            damaged_part = part_mapping.get(damage_type, damage_type)
            
            if damaged_part not in summary["damaged_parts"]:
                summary["damaged_parts"].append(damaged_part)
        
        # Update counts
        summary["dents_count"] += len(image_assessment["dents"])
        summary["scratches_count"] += len(image_assessment["scratches"])
        
        # Update overall score
        summary["total_assessment_score"] = min(
            summary["total_assessment_score"],
            image_assessment["assessment_score"]
        )

def prepare_assessment_response(assessment: Dict) -> Dict:
        """
        Convert all NumPy types to native Python types for JSON serialization
        """
        import numpy as np
        
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

@app.get("/")
async def root():
    return {"message": "Vehicle Inspection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/inspect-vehicle")
async def inspect_vehicle(
    background_tasks: BackgroundTasks,
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...)
):
    """Main endpoint for vehicle inspection"""
    try:
        # Validate file types
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']
        for file in [front_image, back_image, left_image, right_image]:
            if file.content_type not in allowed_types:
                raise HTTPException(400, f"Invalid file type for {file.filename}")
        
        # Create new inspection
        inspection = VehicleInspection()
        
        # Process each image
        images = {
            "front": front_image,
            "back": back_image, 
            "left": left_image,
            "right": right_image
        }
        
        for view_type, image_file in images.items():
            # Save uploaded image
            image_path = f"temp_images/{inspection.car_id}_{view_type}.jpg"
            with open(image_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)
            
            # Process image
            image_assessment = inspection.process_image(image_path, view_type)
            inspection.assessment["images"][view_type] = image_assessment
        
        # Combine license plates from all images
        inspection.combine_license_plates()
        
        # Store assessment (convert to serializable format)
        serializable_assessment = prepare_assessment_response(inspection.assessment)
        assessments_db[inspection.car_id] = serializable_assessment
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, inspection.car_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Vehicle inspection completed successfully",
                #"car_id": inspection.car_id,
                "assessment": serializable_assessment  # Use the serializable version
            }
        )
         
    except Exception as e:
        raise HTTPException(500, f"Inspection failed: {str(e)}")


@app.get("/assessment/{car_id}")
async def get_assessment(car_id: str):
    """Retrieve assessment by car ID"""
    if car_id not in assessments_db:
        raise HTTPException(404, "Assessment not found")
    
    return assessments_db[car_id]

@app.get("/assessments")
async def list_assessments():
    """List all assessments"""
    return assessments_db

async def cleanup_temp_files(car_id: str):
    """Clean up temporary files"""
    import glob
    files = glob.glob(f"temp_images/{car_id}*")
    for file in files:
        try:
            os.remove(file)
        except:
            pass
def find_non_serializable(obj, path=""):
    """Find non-serializable objects in a dictionary"""
    import numpy as np
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            find_non_serializable(value, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            find_non_serializable(item, new_path)
    else:
        if isinstance(obj, (np.ndarray, np.float32, np.float64, np.int32, np.int64)):
            print(f"Non-serializable found at {path}: {type(obj)} - {obj}")

@app.post("/test-detectors")
async def test_detectors(file: UploadFile = File(...)):
    """Test individual detectors to find which one is failing"""
    try:
        # Save uploaded image
        test_path = f"temp_images/test_{uuid.uuid4()}.jpg"
        with open(test_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        image = cv2.imread(test_path)
        results = {}
        
        # Test each detector one by one
        detectors = [
            ("Plate Detector", plate_detector),
            ("Parts Detector", parts_detector),
            ("Damage Detector", damage_detector), 
            ("Dent Detector", dent_detector),
            ("Scratch Detector", scratch_detector)
        ]
        
        for name, detector in detectors:
            try:
                if name == "Parts Detector":
                    result = detector.detect(image, "front")
                else:
                    result = detector.detect(image)
                results[name] = "SUCCESS"
            except Exception as e:
                results[name] = f"FAILED: {str(e)}"
        
        # Cleanup
        os.remove(test_path)
        
        return {"detector_tests": results}
    
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)