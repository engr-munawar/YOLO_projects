import numpy as np
from typing import List, Dict, Tuple

def analyze_car_view_advanced(detections: List[Dict], img_width: int) -> str:
    """
    Advanced car view analysis that handles hybrid views (front/side, back/side).
    """
    parts_present = [det['class_name'] for det in detections]
    
    # Define parts for each view
    front_dominant_parts = ['front_bumper', 'headlight', 'bonnet', 'front_windscreen']
    back_dominant_parts = ['taillight', 'luggage_door', 'rear_bumper', 'rear_windscreen']
    side_parts = ['sidemirror', 'door', 'tyre']
    
    # Count parts for each category
    front_count = sum(1 for part in parts_present if part in front_dominant_parts)
    back_count = sum(1 for part in parts_present if part in back_dominant_parts)
    side_count = sum(1 for part in parts_present if part in side_parts)
    
    print(f"📊 Parts Analysis - Front: {front_count}, Back: {back_count}, Side: {side_count}")
    
    # Rule 1: If we have strong front/back dominance with minimal side parts, it's front/back
    if front_count >= 2 and side_count <= 1:
        print("🎯 Strong front dominance detected")
        return 'front'
    
    if back_count >= 2 and side_count <= 1:
        print("🎯 Strong back dominance detected")
        return 'back'
    
    # Rule 2: If we have both front/back and side parts, determine hybrid view
    if front_count > 1 and side_count >= 1:
        side_direction = determine_side_direction(detections, img_width)
        return f'front_{side_direction}'
    
    if back_count > 1 and side_count >= 1:
        side_direction = determine_side_direction(detections, img_width)
        return f'back_{side_direction}'
    
    # Rule 3: Pure side view
    if side_count > 0:
        side_direction = determine_side_direction(detections, img_width)
        return f'{side_direction}_side'
    
    # Rule 4: Default to front/back based on parts count
    if front_count > back_count:
        return 'front'
    elif back_count > front_count:
        return 'back'
    else:
        # If equal, use spatial distribution
        if detections:
            avg_x = np.mean([det['x_center'] for det in detections])
            return 'front' if avg_x < 0.5 else 'back'
        return 'front'

def determine_side_direction(detections: List[Dict], img_width: int) -> str:
    """
    Determine left/right direction for hybrid views (front/side, back/side).
    Uses the same priority logic as side view determination.
    """
    # Use door position to determine side
    doors = [det for det in detections if det['class_name'] == 'door']
    if doors:
        best_door = max(doors, key=lambda x: x['confidence'])
        door_x_center = best_door['x_center'] * img_width
    
    # Priority 1: Check headlight (for front-side views)
    headlights = [det for det in detections if det['class_name'] == 'headlight']
    if headlights:
        best_headlight = max(headlights, key=lambda x: x['confidence'])
        headlight_x_center = best_headlight['x_center'] * img_width
        return 'left' if headlight_x_center < door_x_center else 'right'
    
    # Priority 2: Check taillight (for back-side views)
    taillights = [det for det in detections if det['class_name'] == 'taillight']
    if taillights:
        best_taillight = max(taillights, key=lambda x: x['confidence'])
        taillight_x_center = best_taillight['x_center'] * img_width
        return 'right' if taillight_x_center < door_x_center else 'left'
    
    # Priority 3: Check sidemirror
    sidemirrors = [det for det in detections if det['class_name'] == 'sidemirror']
    if sidemirrors:
        best_mirror = max(sidemirrors, key=lambda x: x['confidence'])
        mirror_x_center = best_mirror['x_center'] * img_width
        return 'left' if mirror_x_center < door_x_center else 'right'
          
    return 'right'  # Default


def assign_left_right(part_name: str, box, view_type: str, img_width: int):
    """
    Assign left/right based on X center of bounding box.
    Works for headlight, sidemirror, taillight.
    """

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_width

    if view_type == "front":
        return f"{part_name}_right" if cx < 0.5 else f"{part_name}_left"

    elif view_type == "back":
        return f"{part_name}_left" if cx < 0.5 else f"{part_name}_right"

    elif view_type == "left" or view_type == "front_left" or view_type == "back_left":
        return f"{part_name}_left"

    elif view_type == "right" or view_type == "front_right" or view_type == "back_right":
        return f"{part_name}_right"

    return part_name



def assign_door_instance(detections: List[Dict], detection_info: List[Dict], img_width: int, view_type: str) -> str:
    
    # Use door position to determine position
    doors = [det for det in detections if det['class_name'] == 'door']
    if doors:
        door_1 = max(doors, key=lambda x: x['confidence'])
        door_x_center_1 = door_1['x_center'] * img_width
        door_2 = min(doors, key=lambda x: x['confidence'])
        door_x_center_2 = door_2['x_center'] * img_width
        doors_x_center = (door_x_center_1 + door_x_center_2) / 2
        
    
    # Use tyre position to determine position
    tyres = [det for det in detections if det['class_name'] == 'tyre']
    if tyres:
        tyre_1 = max(tyres, key=lambda x: x['confidence'])
        tyre_x_center_1 = tyre_1['x_center'] * img_width
        tyre_2 = min(tyres, key=lambda x: x['confidence'])
        tyre_x_center_2 = tyre_2['x_center'] * img_width
        tyres_x_center = (tyre_x_center_1 + tyre_x_center_2) / 2
        
    
       
    # check door position
        
    if tyres_x_center: # first from tyres position check door position
        if detection_info['class_name'] == 'door' and (view_type == "left" or view_type == "front_left" or view_type == "back_left"):
        
            door_x_center = detection_info['x_center'] * img_width
            return 'door_front_left' if door_x_center < tyres_x_center else 'door_back_left'
    
        if detection_info['class_name'] == 'door' and (view_type == "right" or view_type == "front_right" or view_type == "back_right"):
        
            door_x_center = detection_info['x_center'] * img_width
            return 'door_front_right' if door_x_center > tyres_x_center else 'door_back_right'
    
    else: # door position based on both doors x_center
        if detection_info['class_name'] == 'door' and (view_type == "left" or view_type == "front_left" or view_type == "back_left"):
        
            door_x_center = detection_info['x_center'] * img_width
            return 'door_front_left' if door_x_center < doors_x_center else 'door_back_left'
    
        if detection_info['class_name'] == 'door' and (view_type == "right" or view_type == "front_right" or view_type == "back_right"):
        
            door_x_center = detection_info['x_center'] * img_width
            return 'door_front_right' if door_x_center > doors_x_center else 'door_back_right'
    
    return "door"


def assign_tyre_instance(detections: List[Dict], detection_info: List[Dict], img_width: int, view_type: str) -> str:
    
    # Use door position to determine position
    doors = [det for det in detections if det['class_name'] == 'door']
    if doors:
        best_door = max(doors, key=lambda x: x['confidence'])
        doors_x_center = best_door['x_center'] * img_width
        
    
    # check tyre position
    if detection_info['class_name'] == 'tyre' and (view_type == "left" or view_type == "front_left" or view_type == "back_left"):
        
        tyre_x_center = detection_info['x_center'] * img_width
        return 'tyre_front_left' if tyre_x_center < doors_x_center else 'tyre_back_left'
    
    if detection_info['class_name'] == 'tyre' and (view_type == "right" or view_type == "front_right" or view_type == "back_right"):
        
        tyre_x_center = detection_info['x_center'] * img_width
        return 'tyre_front_right' if tyre_x_center > doors_x_center else 'tyre_back_right'
    
    return "tyre"
