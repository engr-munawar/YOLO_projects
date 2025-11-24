import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import cv2

from models.parts_detector import CarPartsDetector
from models.plate_detector import LicensePlateDetector
from models.damage_detector import DamageDetector
from models.dent_detector import DentDetector
from models.scratch_detector import ScratchDetector
from utils.select_user_car import detect_and_segment_user_car

# Initialize detectors
parts_detector = CarPartsDetector()
plate_detector = LicensePlateDetector()
damage_detector = DamageDetector()
damage_detector = DamageDetector()
dent_detector = DentDetector()
scratch_detector = ScratchDetector()

class CarInspection:
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
            "uploaded_view": view_type,
            "detected_view": parts_assessment.get("detected_view"),
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
            
            print(f"Successfully processed {view_type} ({image_assessment['detected_view']}) image with plate: {combined_text}")
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
        print(f"=== DEBUG: All detected parts from ALL views ===")
        #print(self.assessment["images"])
        for part in sorted(actually_detected_parts):
            print(f"  - {part}")
        print(f"Total unique parts detected: {len(actually_detected_parts)}")
        print(f"=================================================")

        # Second pass: collect missing parts, damages, etc.
        for view_type, view_data in self.assessment["images"].items():
            # Skip views that had errors
            if "error" in view_data:
                print(f"Skipping {view_type} view due to error: {view_data['error']}")
                continue
            # Collect missing parts
            if "missing_parts" in view_data:
                all_reported_missing_parts.extend(view_data["missing_parts"])
        
            
            # Collect damaged parts
            if "damages" in view_data:
                for damage in view_data["damages"]:
                    part_name = damage.get("part", "")
                    #part_name = damage_to_part_mapping.get(damage_type)
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
        
        print(f"=== DEBUG: All reported missing parts ===")
        for part in sorted(set(all_reported_missing_parts)):
            print(f"  - {part}")
        print(f"=========================================")

        # Reconcile: A part is only truly missing if:
        # 1. It's reported as missing AND
        # 2. It's not detected in ANY view AND  
        # 3. It's not listed as damaged
        truly_missing_parts = []
        for part in set(all_reported_missing_parts):
            if (part not in actually_detected_parts and 
                part not in all_damaged_parts):
                truly_missing_parts.append(part)
        print(f"=== DEBUG: Final missing parts after reconciliation ===")
        for part in sorted(truly_missing_parts):
            print(f"  - {part}")
        print(f"======================================================")

        # Update summary
        summary["missing_parts"] = truly_missing_parts
        summary["damaged_parts"] = all_damaged_parts
        summary["dents_count"] = total_dents
        summary["scratches_count"] = total_scratches
        
        # Calculate overall score (average of view scores)
        if view_scores:
            summary["total_assessment_score"] = sum(view_scores) // len(view_scores)