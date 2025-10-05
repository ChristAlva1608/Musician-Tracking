from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Optional, List, Dict, Any, Tuple


class YOLOFaceDetector:
    """YOLO face detection model focused on bounding box detection"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        self.confidence = confidence

        # Use absolute path to checkpoints
        if model_path is None:
            # Get the src directory (3 levels up from this file)
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            model_path = os.path.join(src_dir, "checkpoints", "yolov8n-face.pt")

        self.model_path = model_path
        self.model_type = "face_detection"
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO face model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO face model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect faces in the frame using YOLO
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            YOLO detection results or None if detection fails
        """
        if self.model is None:
            return None
            
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            return results
        except Exception as e:
            print(f"❌ YOLO face detection failed: {e}")
            return None
    
    def extract_bboxes(self, results: Any, frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes from YOLO detection results
        
        Args:
            results: YOLO detection results
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            List of bounding box dictionaries with keys: x, y, w, h, confidence
        """
        bboxes = []
        
        if not results or len(results) == 0:
            return bboxes
        
        try:
            h, w = frame_shape[:2]
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Extract boxes in xyxy format
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        
                        # Convert to x, y, w, h format
                        x = int(x1)
                        y = int(y1)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        
                        # Ensure coordinates are within bounds
                        x = max(0, min(x, w))
                        y = max(0, min(y, h))
                        width = max(1, min(width, w - x))
                        height = max(1, min(height, h - y))
                        
                        bboxes.append({
                            'x': x,
                            'y': y,
                            'w': width,
                            'h': height,
                            'confidence': float(conf),
                            'x1': x,
                            'y1': y,
                            'x2': x + width,
                            'y2': y + height
                        })
                        
        except Exception as e:
            print(f"❌ Error extracting YOLO bboxes: {e}")
        
        return bboxes
    
    def extract_bboxes_with_pad(self, results: Any, frame_shape: Tuple[int, int, int], 
                               padding: int = 30) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes with padding from YOLO detection results
        
        Args:
            results: YOLO detection results
            frame_shape: Shape of the input frame (height, width, channels)
            padding: Padding to add around each bounding box in pixels
            
        Returns:
            List of bounding box dictionaries with both original and padded coordinates
        """
        bboxes = []
        original_bboxes = self.extract_bboxes(results, frame_shape)
        
        if not original_bboxes:
            return bboxes
        
        try:
            h, w = frame_shape[:2]
            
            for bbox in original_bboxes:
                # Original coordinates
                x, y, width, height = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                
                # Calculate padded coordinates
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(w - x_pad, width + 2 * padding)
                h_pad = min(h - y_pad, height + 2 * padding)
                
                # Create enhanced bbox dictionary
                padded_bbox = bbox.copy()
                padded_bbox.update({
                    'x_pad': x_pad,
                    'y_pad': y_pad,
                    'w_pad': w_pad,
                    'h_pad': h_pad,
                    'x1_pad': x_pad,
                    'y1_pad': y_pad,
                    'x2_pad': x_pad + w_pad,
                    'y2_pad': y_pad + h_pad,
                    'padding': padding
                })
                
                bboxes.append(padded_bbox)
                
        except Exception as e:
            print(f"❌ Error adding padding to bboxes: {e}")
        
        return bboxes
    
    def draw_bboxes(self, frame: np.ndarray, results: Any, 
                   draw_padding: bool = False, padding: int = 30) -> np.ndarray:
        """
        Draw bounding boxes on the frame
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            draw_padding: Whether to draw padded bounding boxes
            padding: Padding amount for padded boxes
            
        Returns:
            Frame with drawn bounding boxes
        """
        if not results or len(results) == 0:
            return frame
        
        try:
            # Get bounding boxes
            if draw_padding:
                bboxes = self.extract_bboxes_with_pad(results, frame.shape, padding)
            else:
                bboxes = self.extract_bboxes(results, frame.shape)
            
            for i, bbox in enumerate(bboxes):
                # Draw padded bounding box (yellow)
                if draw_padding and 'x_pad' in bbox:
                    x_pad, y_pad = bbox['x_pad'], bbox['y_pad']
                    w_pad, h_pad = bbox['w_pad'], bbox['h_pad']
                    
                    cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), 
                                (0, 255, 255), 2)
                    # cv2.putText(frame, f"Padded (+{padding}px)", (x_pad, y_pad - 5), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    # Draw bounding box (red)
                    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                    conf = bbox['confidence']
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    
                    # Draw confidence label
                    label = f"Face {i+1}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        except Exception as e:
            print(f"❌ Error drawing YOLO face bboxes: {e}")
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "YOLO Face Detector",
            "type": "face_detection",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "available": self.model is not None,
            "functions": ["detect", "extract_bboxes", "extract_bboxes_with_pad", "draw_bboxes"]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # YOLO models don't need explicit cleanup
        pass