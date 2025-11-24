import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

# ==================== CAR DETECTION AND SEGMENTATION ====================

# Load YOLO11 segmentation model for car detection
car_model = YOLO("yolo11s-seg.pt")  # Segmentation model
CONFIDENCE_THRESHOLD = 0.3 # reduced to include bad angled cars

def select_user_vehicle(masks, boxes, confidences, img_height, img_width=None):
    """Select the user's vehicle from detected cars using segmentation."""
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return 0

    best_index = 0
    best_score = -1
    valid_cars_found = False

    for i, (mask, box, conf) in enumerate(zip(masks, boxes, confidences)):
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        area = w * h
        
        # Calculate area as percentage of image area for better normalization
        if img_width is not None:
            img_area = img_width * img_height
            area_ratio = area / img_area
            print(f"Car {i}: Area={area}, Area Ratio={area_ratio:.3f}, Conf={conf:.3f}")
            
            # Use relative area threshold instead of absolute
            if area_ratio < 0.15:  # Skip if car occupies less than 15% of image
                print(f"  Skipping - too small (ratio: {area_ratio:.3f})")
                continue
        else:
            # Fallback to absolute area for backward compatibility
            print(f"Car {i}: Area={area}, Conf={conf:.3f}")
            if area < 15000:  # Increased threshold for original images
                print(f"  Skipping - too small (area: {area})")
                continue
        
        valid_cars_found = True
        bottom = y2
        vertical_score = bottom / img_height
        score = conf * area * vertical_score

        print(f"  Valid car - Vertical Score: {vertical_score:.3f}, Final Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_index = i

    if not valid_cars_found:
        print("❌ No cars passed the size and confidence filters")
        return None
        
    return best_index

def detect_and_segment_user_car(input_image_path):
    """Detect and segment user's car, return masked image without surroundings."""
    
    # First try to get a good crop using object detection
    img, shape, detection_success = detect_and_crop_user_car_fallback(input_image_path)
    
    fallback_result = "user_car.jpg"
    cv2.imwrite(fallback_result, img)
    print(f"✅ Saved car detection result as {fallback_result}, Shape: {shape}")

    # If object detector could not find a proper car, return original image
    if not detection_success:
        print("❌ Car detection failed, no suitable car found.")
        return img

    # Load image
    if img is None:
        print("Error: Could not load image.")
        return None

    img_height, img_width = img.shape[:2]
    print(f"Processing image of size: {img_width}x{img_height}")

    # Run YOLO11 segmentation inference for car detection
    results = car_model(img, classes=[2])[0]

    # Check if we have segmentation results
    if results.masks is None:
        print("❌ No segmentation masks found. Returning detected car crop.")
        return img

    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    print(f"✅ Detected {len(boxes)} car(s) with segmentation masks")

    # Select the most likely user car
    user_idx = select_user_vehicle(masks, boxes, confidences, img_height, img_width)

    # If no suitable car found in segmentation, return the detection crop
    if user_idx is None:
        print("❌ No suitable car found in segmentation, returning detection crop.")
        return img

    # Create visualization for car detection
    annotated_img = img.copy()
    user_car_mask = None
    user_car_bbox = None
    
    for i, (mask, box, conf) in enumerate(zip(masks, boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if i == user_idx else (0, 255, 0)
        label = f"Your Car? {conf:.2f}" if i == user_idx else f"Other Car {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, y1 +15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw segmentation mask (semi-transparent)
        if i == user_idx:
            user_car_mask = mask
            user_car_bbox = (x1, y1, x2, y2)
            
            # Create colored mask for visualization
            colored_mask = np.zeros_like(img)
            colored_mask[:, :] = color
            mask_resized = cv2.resize(mask, (img_width, img_height))
            mask_bool = mask_resized > 0.5
            annotated_img[mask_bool] = cv2.addWeighted(annotated_img[mask_bool], 0.7, colored_mask[mask_bool], 0.3, 0)

    # Create masked image containing ONLY the user's car
    if user_car_mask is not None:
        # Resize mask to original image size
        mask_resized = cv2.resize(user_car_mask, (img_width, img_height))
        mask_bool = mask_resized > 0.5
        
        # Create masked image (set background to black or white)
        masked_img = img.copy()
        # Option 1: Set background to black (better for parts detection)
        masked_img[~mask_bool] = [0, 0, 0]
                
        # Crop to bounding box with some padding to include entire mask
        x1, y1, x2, y2 = user_car_bbox
        padding = 10  # Add some padding to ensure entire mask is included
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        segmented_crop = masked_img[y1:y2, x1:x2]
        
        segmented_path = "user_car_segmented.jpg"
        cv2.imwrite(segmented_path, segmented_crop)
        print(f"✅ Saved segmented user's car as {segmented_path}")
        
        # Also save the mask visualization
        mask_visual = np.zeros_like(img)
        mask_visual[mask_bool] = img[mask_bool]
        return mask_visual
    else:
        print("❌ No suitable car detected with segmentation")
        return img

def detect_and_crop_user_car_fallback(input_image_path):
    """Fallback method using bounding boxes if segmentation fails."""
    
    # Load YOLO detection model for fallback
    det_model = YOLO("yolo11s.pt")
    
    img = cv2.imread(input_image_path)
    if img is None:
        return None, None, False
    
    original_img = img.copy()
    img_height, img_width = img.shape[:2]
    print(f"Original image size: {img_width}x{img_height}")

    # Run YOLO detection inference
    results = det_model(img, classes=[2], conf=CONFIDENCE_THRESHOLD)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    print(f"Fallback detection found {len(boxes)} cars")

    # Select the most likely user car - using original image dimensions
    user_idx = select_user_vehicle([None] * len(boxes), boxes, confidences, img_height, img_width)

    # If no suitable car found, return original image
    if user_idx is None:
        print("❌ No suitable car found in fallback detection")
        return original_img, original_img.shape, False

    # Create visualization
    annotated_img = img.copy()
    user_car_bbox = None
    
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if i == user_idx else (0, 255, 0)
        label = f"Your Car? {conf:.2f}" if i == user_idx else f"Other Car {conf:.2f}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, y1 +15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if i == user_idx:
            user_car_bbox = (x1, y1, x2, y2)

    # Crop user's vehicle
    if user_idx is not None:
        x1, y1, x2, y2 = user_car_bbox
        car_crop = img[y1:y2, x1:x2]
        print(f"✅ Cropped car size: {car_crop.shape}")
        return car_crop, car_crop.shape, True
    else:
        print("❌ No suitable car detected in fallback")
        return original_img, original_img.shape, False
    
if __name__ == "__main__":
    result = detect_and_segment_user_car("E:/AI/CV/Task by Hasnain sb/vehicle inspection project/test images/c_65.jpg")
    if result is not None:
        print(f"✅ Final result shape: {result.shape}")
    else:
        print("❌ Final result is None")