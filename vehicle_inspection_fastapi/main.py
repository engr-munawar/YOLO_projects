from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os

from utils.car_inspection import CarInspection
from utils.prepare_assessment_responce import prepare_assessment_response
from utils.json_storage import load_assessments, save_assessments
from utils.cleanup_temp_folder import cleanup_temp_folder



app = FastAPI(title="Vehicle Inspection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Create temp directory
os.makedirs("temp_images", exist_ok=True)

# assessment.json file storage for assessments (in production, use database)
assessments_db = load_assessments()


@app.get("/")
async def root():
    return {"message": "Vehicle Inspection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/inspect-vehicle")
async def inspect_vehicle(    
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
    # 🧹 CLEANUP FIRST: Completely empty temp folder before starting new inspection
    cleanup_temp_folder()
    
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

    inspection = CarInspection()

    # Save + process required images
    for view, file in REQUIRED.items():
        if file is None:
            raise HTTPException(400, f"Missing required image: {view}")

        save_path = f"temp_images/{inspection.car_id}_{view}.jpg"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        assessment = inspection.process_image(save_path, view)
        inspection.assessment["images"][view] = assessment
        # Update summary after storing the image so the summary sees this view
        inspection._update_summary(None)

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
            inspection.assessment['images'][view] = assessment
            # Update summary after storing optional view
            inspection._update_summary(None)
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
    save_assessments(assessments_db)

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


# @app.post("/test-detectors")
# async def test_detectors(file: UploadFile = File(...)):
#     """Test individual detectors to find which one is failing"""
#     try:
#         # Save uploaded image
#         test_path = f"temp_images/test_{uuid.uuid4()}.jpg"
#         with open(test_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
        
#         image = cv2.imread(test_path)
#         results = {}
        
#         # Test each detector one by one
#         detectors = [
#             ("Plate Detector", plate_detector),
#             ("Parts Detector", parts_detector),
#             ("Damage Detector", damage_detector), 
#             ("Dent Detector", dent_detector),
#             ("Scratch Detector", scratch_detector)
#         ]
        
#         for name, detector in detectors:
#             try:
#                 if name == "Parts Detector":
#                     result = detector.detect(image, "front")
#                 else:
#                     result = detector.detect(image)
#                 results[name] = "SUCCESS"
#             except Exception as e:
#                 results[name] = f"FAILED: {str(e)}"
        
#         # Cleanup
#         os.remove(test_path)
        
#         return {"detector_tests": results}
    
#     except Exception as e:
#         return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)