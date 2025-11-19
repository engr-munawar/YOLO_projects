import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Dict
from thefuzz import fuzz, process

class LicensePlateDetector:
    def __init__(self, model_path: str = "models/plate_detector.pt"):
        print("Loading YOLO plate detector...")
        self.plate_detector = YOLO(model_path)
        
        print("Loading PaddleOCR...")
        self.ocr_reader = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
        )
    
    def put_text_pil(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                    color: Tuple[int, int, int] = (0, 0, 255), font_size: int = 24) -> np.ndarray:
        """Draw text using PIL to handle special characters properly"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text(pos, text, fill=color, font=font)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def parse_ocr_result(self, ocr_result) -> List[Dict]:
        """Parse PaddleOCR result"""
        lines = []
        
        if not ocr_result:
            return lines
        
        try:
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                result_dict = ocr_result[0]
                
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                dt_polys = result_dict.get('dt_polys', [])
                
                print(f"  📊 Found {len(rec_texts)} text elements in OCR result")
                
                for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    if i < len(dt_polys):
                        coordinates = dt_polys[i]
                        if hasattr(coordinates, 'tolist'):
                            coordinates = coordinates.tolist()
                    else:
                        coordinates = []
                    
                    lines.append({
                        'coordinates': coordinates,
                        'text': str(text),
                        'confidence': float(score)
                    })
                    
                    print(f"    Text {i}: '{text}' (score: {score:.4f})")
                    
        except Exception as e:
            print(f"  ❌ Error parsing OCR format: {e}")
        
        return lines
    
    def has_valid_coordinates(self, coordinates) -> bool:
        """Check if coordinates are valid and non-empty"""
        if coordinates is None:
            return False
        if hasattr(coordinates, '__len__') and len(coordinates) == 0:
            return False
        if hasattr(coordinates, 'size') and coordinates.size == 0:
            return False
        return True
    
    def detect_single_image(self, image: np.ndarray) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Detect license plate from a single image"""
        if image is None:
            return [], None
        
        print("🔍 Detecting license plate...")
        results = self.plate_detector(image, conf=0.5)
        
        if not results[0].boxes:
            print("  ⚠️  No plate detected")
            return [], None

        # Get highest-confidence plate
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        x1, y1, x2, y2 = map(int, boxes[best_idx])
        detection_confidence = float(confs[best_idx])
        
        # Crop plate
        plate_img = image[y1:y2, x1:x2]
        
        print(f"  ✅ Cropped plate: {plate_img.shape}")
        
        print("🔤 Running OCR on plate...")
        try:
            ocr_result = self.ocr_reader.predict(plate_img)
        except Exception as e:
            print(f"  ❌ OCR failed: {e}")
            ocr_result = None
        
        license_lines = []
        annotated = image.copy()
        
        # Draw plate bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if not ocr_result:
            print("  ⚠️  OCR returned no result")
            return [], annotated
        
        # Parse OCR results
        ocr_lines = self.parse_ocr_result(ocr_result)
        print(f"  📝 Parsed {len(ocr_lines)} OCR lines")
        
        line_number = 0
        
        for ocr_line in ocr_lines:
            text = ocr_line['text'].strip()
            confidence = ocr_line['confidence']
            coordinates = ocr_line['coordinates']
            
            if not text:
                continue
            
            # Auto-correct the text
            line_type = self._classify_line(text)
            corrected_text = self.correct_license_text(text, line_type)
                
            if confidence > 0.5:
                # Convert coordinates
                abs_bbox = []
                if self.has_valid_coordinates(coordinates):
                    for point in coordinates:
                        if hasattr(point, '__len__') and len(point) >= 2:
                            point_x = point[0] if hasattr(point, '__getitem__') else float(point)
                            point_y = point[1] if hasattr(point, '__getitem__') else float(point)
                            
                            abs_x = int(x1 + point_x * (x2 - x1) / plate_img.shape[1])
                            abs_y = int(y1 + point_y * (y2 - y1) / plate_img.shape[0])
                            abs_bbox.append((abs_x, abs_y))
                
                # Default coordinates if none available
                if not abs_bbox:
                    line_height = 30
                    abs_bbox = [
                        (x1, y1 + line_number * line_height),
                        (x2, y1 + line_number * line_height),
                        (x2, y1 + (line_number + 1) * line_height),
                        (x1, y1 + (line_number + 1) * line_height)
                    ]
                
                # Calculate average Y position
                avg_y = sum(point[1] for point in abs_bbox) / len(abs_bbox) if abs_bbox else line_number * 30
                
                license_lines.append({
                    'text': corrected_text,
                    'original_text': text,
                    'line_type': line_type,
                    'confidence': float(confidence),
                    'position': line_number,
                    'line_bbox': abs_bbox,
                    'avg_y': avg_y,
                    'detection_confidence': detection_confidence,
                    'combined_confidence': detection_confidence * confidence
                })
                
                print(f"    ✅ Line {line_number}: '{corrected_text}' (conf: {confidence:.4f})")
                
                # Draw on annotated image
                if abs_bbox and len(abs_bbox) >= 4:
                    try:
                        abs_bbox_np = np.array(abs_bbox, dtype=np.int32)
                        cv2.polylines(annotated, [abs_bbox_np], True, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"    ⚠️  Could not draw bounding box: {e}")
                
                # Draw text
                if abs_bbox:
                    text_x = min(point[0] for point in abs_bbox)
                    text_y = min(point[1] for point in abs_bbox) - 5
                else:
                    text_x = x1
                    text_y = y1 + line_number * 30 - 5
                    
                try:
                    annotated = self.put_text_pil(annotated, f"{corrected_text} ({confidence:.2f})", 
                                                (text_x, text_y), (0, 255, 0), 16)
                except Exception as e:
                    print(f"    ⚠️  Could not draw text: {e}")
                
                line_number += 1
            else:
                print(f"    ⚠️  Skipped (low confidence): '{text}' (conf: {confidence:.4f})")
        
        # Sort lines by position
        license_lines.sort(key=lambda x: x['avg_y'])
        
        if not license_lines:
            print("  ❌ No valid text detected by OCR")
        
        return license_lines, annotated
    
    def detect_multiple_images(self, images: List[np.ndarray]) -> Tuple[List[Dict], np.ndarray]:
        """Process multiple images and combine results for best license plate"""
        all_detections = []
        
        for i, img in enumerate(images):
            if img is None:
                continue
                
            print(f"\n📸 Processing image {i+1}...")
            license_lines, annotated_img = self.detect_single_image(img)
            
            if license_lines:
                for line in license_lines:
                    line['source_image'] = i
                    line['annotated_img'] = annotated_img
                all_detections.extend(license_lines)
                print(f"  ✅ Found {len(license_lines)} lines in image {i+1}")
            else:
                print(f"  ⚠️  No license plate lines found in image {i+1}")
        
        # Get best combined result from all images
        best_lines = self._get_best_combined_result(all_detections)
        
        # Get best annotated image
        best_annotated_img = self._get_best_annotated_image(all_detections, images)
        
        return best_lines, best_annotated_img
    
    def _get_best_combined_result(self, all_detections: List[Dict]) -> List[Dict]:
        """Combine results from all images to get the most complete license plate"""
        if not all_detections:
            return []
        
        print(f"\n🔄 Combining results from {len(set(d['source_image'] for d in all_detections))} images...")
        
        # Group by image
        image_groups = {}
        for detection in all_detections:
            img_idx = detection['source_image']
            if img_idx not in image_groups:
                image_groups[img_idx] = []
            image_groups[img_idx].append(detection)
        
        # Score each image based on completeness
        best_image_idx = None
        best_score = -1
        
        for img_idx, lines in image_groups.items():
            score = self._calculate_completeness_score(lines)
            print(f"  Image {img_idx+1} completeness score: {score}")
            
            if score > best_score:
                best_score = score
                best_image_idx = img_idx
        
        if best_image_idx is not None:
            best_lines = image_groups[best_image_idx]
            print(f"  🏆 Selected Image {best_image_idx+1} as most complete")
            
            # Sort lines properly
            best_lines.sort(key=lambda x: (0 if x['line_type'] == 'Number Plate' else 1, x['position']))
            return best_lines
        
        return []
    
    def _calculate_completeness_score(self, lines: List[Dict]) -> int:
        """Calculate how complete a license plate detection is"""
        score = 0
        has_vehicle_code = False
        has_registration = False
        has_city = False
        
        for line in lines:
            text = line['text'].upper()
            line_type = line['line_type']
            
            if line_type == "Number Plate":
                # Check for vehicle code (mostly letters)
                if sum(c.isalpha() for c in text) >= 2:
                    has_vehicle_code = True
                    score += 3
                # Check for registration (has numbers)
                if any(c.isdigit() for c in text):
                    has_registration = True
                    score += 3
                # Bonus for complete number plates
                if any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
                    score += 2
                    
            elif line_type == "City":
                has_city = True
                score += 2
        
        # Bonus for complete license plate
        if has_vehicle_code and has_registration:
            score += 5
        if has_city:
            score += 3
        if len(lines) >= 2:
            score += 2
            
        return score
    
    def _get_best_annotated_image(self, all_detections: List[Dict], images: List[np.ndarray]) -> np.ndarray:
        """Get the best annotated image"""
        if all_detections:
            image_scores = {}
            for detection in all_detections:
                img_idx = detection['source_image']
                if img_idx not in image_scores:
                    image_scores[img_idx] = 0
                image_scores[img_idx] += detection['combined_confidence']
            
            if image_scores:
                best_img_idx = max(image_scores.keys(), key=lambda x: image_scores[x])
                for detection in all_detections:
                    if detection['source_image'] == best_img_idx:
                        return detection['annotated_img']
        
        # Fallback
        if images and images[0] is not None:
            fallback_img = images[0].copy()
        else:
            fallback_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return self.put_text_pil(fallback_img, "NO_PLATE_DETECTED", (50, 50), (0, 0, 255), 24)
    
    def detect(self, image: np.ndarray) -> str:
        """Simple detection for single image"""
        license_lines, _ = self.detect_single_image(image)
        if not license_lines:
            return "NO_PLATE_DETECTED"
        return " ".join(line['text'] for line in license_lines)
    
    def correct_license_text(self, text: str, line_type: str) -> str:
        """Auto-correct license plate text"""
        original_text = text
        text_upper = text.upper().strip()
        
        if line_type == "City":
            PAKISTAN_CITIES = [
                'ISLAMABAD', 'LAHORE', 'KARACHI', 'PESHAWAR', 'QUETTA',
                'PUNJAB', 'SINDH', 'KPK', 'BALOCHISTAN',
                'ICT-ISLAMABAD', 'PCT-LAHORE', 'SCT-KARACHI', 'KCT-PESHAWAR'
            ]
            
            # Common corrections
            COMMON_CORRECTIONS = {
                'ISLANABAD': 'ISLAMABAD', 'ISLAMABD': 'ISLAMABAD', 'PUNJB': 'PUNJAB',
                'SIND': 'SINDH', 'KARACH': 'KARACHI', 'LAHOR': 'LAHORE'
            }
            
            for wrong, correct in COMMON_CORRECTIONS.items():
                if wrong in text_upper:
                    corrected = text_upper.replace(wrong, correct)
                    print(f"    🔧 Exact correction: '{original_text}' → '{corrected}'")
                    return corrected
            
            # Fuzzy matching
            best_match, score = process.extractOne(text_upper, PAKISTAN_CITIES, scorer=fuzz.partial_ratio)
            
            if score >= 70:
                if text_upper.startswith(('C-', 'ICT-', 'P-', 'PCT-', 'S-', 'SCT-', 'K-', 'KCT-')):
                    prefix = text_upper.split('-')[0] + '-'
                    if prefix in ['C-', 'ICT-']:
                        corrected = 'ICT-ISLAMABAD'
                    elif prefix in ['P-', 'PCT-']:
                        corrected = 'PCT-LAHORE'
                    elif prefix in ['S-', 'SCT-']:
                        corrected = 'SCT-KARACHI'
                    elif prefix in ['K-', 'KCT-']:
                        corrected = 'KCT-PESHAWAR'
                    else:
                        corrected = best_match
                else:
                    corrected = best_match
                
                if original_text != corrected:
                    print(f"    🔧 Fuzzy match: '{original_text}' → '{corrected}' (score: {score})")
                return corrected
        
        return original_text
    
    def _classify_line(self, text: str) -> str:
        """Classify line as number plate or city name"""
        clean_text = text.upper().replace('-', '').replace(' ', '')
        
        city_keywords = ['ISLAMABAD', 'LAHORE', 'KARACHI', 'PUNJAB', 'SINDH', 'KPK', 'BALOCHISTAN']
        if any(keyword in clean_text for keyword in city_keywords):
            return "City"
        
        alphanumeric_chars = [c for c in clean_text if c.isalnum()]
        if 3 <= len(alphanumeric_chars) <= 8 and any(c.isdigit() for c in clean_text):
            return "Number Plate"
        
        if len(clean_text) > 8:
            return "City"
        else:
            return "Number Plate"
    
    def get_detailed_detection(self, image: np.ndarray) -> Dict:
        """Get detailed detection results"""
        license_lines, annotated_img = self.detect_single_image(image)
        
        # Convert image to base64
        import base64
        import io
        from PIL import Image
        
        annotated_img_base64 = None
        if annotated_img is not None:
            img_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            annotated_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'lines': license_lines,
            'annotated_image': annotated_img_base64,
            'combined_text': " ".join(line['text'] for line in license_lines) if license_lines else "NO_PLATE_DETECTED",
            'total_lines': len(license_lines)
        }

