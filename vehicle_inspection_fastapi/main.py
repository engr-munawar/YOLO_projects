from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
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
            print(f"\nProcessing {view_type} image: {image_path}")
            
            # Read and validate image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "view_type": view_type,
                    "error": f"Could not read image at {image_path}",
                    "parts_detected": [],
                    "missing_parts": [],
                    "damages": [],
                    "dents": [],
                    "scratches": [],
                    "assessment_score": 0,
                    "license_plate": {"combined_text": "ERROR"}
                }
            print(f"Image loaded successfully, shape: {image.shape}")
            
            # Segment the car first
            print("Starting car segmentation...")
            segmented_image = detect_and_segment_user_car(image_path)
            if segmented_image is None:
                return {
                "view_type": view_type,
                "error": "Could not detect car in image",
                "parts_detected": [],
                "missing_parts": [],
                "damages": [],
                "dents": [],
                "scratches": [],
                "assessment_score": 0,
                "license_plate": {"combined_text": "NO_PLATE_DETECTED"}
                }
            
            print(f"Segmentation successful, segmented image shape: {segmented_image.shape}")
            
            # Save segmented image
            seg_path = f"temp_images/segmented_{self.car_id}_{view_type}.jpg"
            cv2.imwrite(seg_path, segmented_image)
            
            # Run plate detection
            print("Running plate detector...")
            plate_result = plate_detector.get_detailed_detection(segmented_image)
            combined_text = plate_result.get('combined_text', 'NO_PLATE_DETECTED')
            print(f"Plate detection result: {combined_text}")
            
            # Run other detectors, first parts detector if view is valid then other detectors
            parts_assessment = parts_detector.detect(segmented_image, view_type)
            # Check if parts detection returned a view validation error
            # If view validation fails - DO NOT STOP PIPELINE
            if parts_assessment.get("error"):
                print(f"❌ View validation failed for {view_type}")

                return {
                    "view_type": view_type,
                    "uploaded_view": f"{view_type}_side",
                    "detected_view": parts_assessment.get("detected_view", "unknown"),
                    "license_plate": {"combined_text": "VIEW_VALIDATION_FAILED"},
                    "parts_detected": [],
                    "missing_parts": parts_assessment.get("missing_parts", []),
                    "damages": [],
                    "dents": [],
                    "scratches": [],
                    "assessment_score": 0,
                    "error": parts_assessment.get("error", "Invalid image view"),
                }
            
            # If view is valid, continue with other detectors

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
            "view_type": parts_assessment.get("uploaded_view"),
            "license_plate": plate_data,
            "parts_detected": parts_assessment.get("detected_parts", []),
            "missing_parts": parts_assessment.get("missing_parts", []),
            "damages": damage_assessment.get("damages", []),
            "dents": dent_assessment.get("dents", []),
            "scratches": scratch_assessment.get("scratches", []),
            "assessment_score": self._calculate_image_score(
                parts_assessment,
                damage_assessment,
                dent_assessment,
                scratch_assessment
            )
            }
            
            # Update summary
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
        
        # Update damaged parts with proper mapping
        damage_to_part_mapping = {
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
        # Track parts that are detected anywhere (not damaged)
        actually_detected_parts = set()
        all_reported_missing_parts = []
        all_damaged_parts = []
        total_dents = 0
        total_scratches = 0
        view_scores = []
        
        # First pass: collect all detected parts from parts_detected
        for view_type, view_data in self.assessment["images"].items():
            if "parts_detected" in view_data:
                for part in view_data["parts_detected"]:
                    part_name = part.get("part_name")
                    if part_name:
                        actually_detected_parts.add(part_name)
        
        # Second pass: collect missing parts, damages, etc.
        for view_type, view_data in self.assessment["images"].items():
            # Collect missing parts
            if "missing_parts" in view_data:
                all_reported_missing_parts.extend(view_data["missing_parts"])
            
            # Collect damaged parts
            if "damages" in view_data:
                for damage in view_data["damages"]:
                    damage_type = damage.get("damage_type", "")
                    part_name = damage_to_part_mapping.get(damage_type)
                    if part_name and part_name not in all_damaged_parts:
                        all_damaged_parts.append(part_name)
            
            # Count dents and scratches
            if "dents" in view_data:
                total_dents += len(view_data["dents"])
            if "scratches" in view_data:
                total_scratches += len(view_data["scratches"])
            
            # Collect assessment scores
            if "assessment_score" in view_data:
                view_scores.append(view_data["assessment_score"])
        
        # Reconcile: A part is only truly missing if:
        # 1. It's reported as missing AND
        # 2. It's not detected in ANY view AND  
        # 3. It's not listed as damaged
        truly_missing_parts = []
        for part in set(all_reported_missing_parts):
            if (part not in actually_detected_parts and 
                part not in all_damaged_parts):
                truly_missing_parts.append(part)
        
        # Update summary
        summary["missing_parts"] = truly_missing_parts
        summary["damaged_parts"] = all_damaged_parts
        summary["dents_count"] = total_dents
        summary["scratches_count"] = total_scratches
        
        # Calculate overall score (average of view scores)
        if view_scores:
            summary["total_assessment_score"] = sum(view_scores) // len(view_scores)

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

from fastapi import UploadFile, File, BackgroundTasks, HTTPException

@app.post("/inspect-vehicle")
async def inspect_vehicle(
    background_tasks: BackgroundTasks,
    
    # Required images
    front: UploadFile = File(..., description="Front view image (REQUIRED)"),
    back: UploadFile = File(..., description="Back view image (REQUIRED)"),
    left: UploadFile = File(..., description="Left view image (REQUIRED)"),
    right: UploadFile = File(..., description="Right view image (REQUIRED)"),
    # optional images
    angle_1: Union[UploadFile, None, str] = File(None), # without union we have to manually uncheck optional image if don't upload the image otherwise it throws error
    angle_2: Union[UploadFile, None, str] = File(None),
    angle_3: Union[UploadFile, None, str] = File(None)
):
    
    REQUIRED = {
        "front": front,
        "back": back,
        "left": left,
        "right": right
    }

    OPTIONAL = {
        "angle_1": angle_1,
        "angle_2": angle_2,
        "angle_3": angle_3
    }

    inspection = VehicleInspection()

    # Save + process required images
    for view, file in REQUIRED.items():
        if file is None:
            raise HTTPException(400, f"Missing required image: {view}")

        save_path = f"temp_images/{inspection.car_id}_{view}.jpg"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        assessment = inspection.process_image(save_path, view)
        inspection.assessment["images"][view] = assessment

    # --- Process optional views safely ---
    
    for view, file in OPTIONAL.items():
        # Skip if no file was provided for this optional view
        if file is None:
            print(f"  ⚠️ Optional view {view} not provided; skipping")
            continue

        # Some clients may send an explicit string like "none" or "false"
        if isinstance(file, str) and file.strip().lower() in ("", "none", "false"):
            print(f"  ⚠️ Optional view {view} value indicates no file; skipping")
            continue

        try:
            save_path = f"temp_images/{inspection.car_id}_{view}.jpg"

            # If we received an UploadFile-like object, read and save its contents
            if hasattr(file, "read"):
                content = await file.read()
                with open(save_path, "wb") as f:
                    f.write(content)
            else:
                # Unknown type for optional file — skip and log
                print(f"  ⚠️ Skipping optional view {view} - unsupported type {type(file)}")
                continue

            assessment = inspection.process_image(save_path, view)
            inspection.assessment[view] = assessment
            print(f"✅ Processed optional view: {view}")

        except Exception as e:
            print(f"⚠️ Error processing optional view {view}: {e}")
            inspection.assessment[view] = {
                "view_type": view,
                "error": str(e)
            }


    # Combine plates
    inspection.combine_license_plates()

    # Convert result for JSON
    final = prepare_assessment_response(inspection.assessment)
    assessments_db[inspection.car_id] = final

    # Cleanup
    background_tasks.add_task(cleanup_temp_files, inspection.car_id)

    return {
        "message": "Inspection completed successfully",
        #"car_id": inspection.car_id,
        "uploaded_images": {
            "required": list(REQUIRED.keys()),
            "optional": [v for v, f in OPTIONAL.items() if f is not None]
        },
        "assessment": final
    }


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
            print(f"🧹 Cleaned up temporary file: {file}")
        except Exception as e:
            print(f"⚠️  Could not delete {file}: {e}")

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