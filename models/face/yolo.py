from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .base_face_detector import BaseFaceDetector


class YOLOFaceDetector(BaseFaceDetector):
    """YOLO face detection model"""
    
    def __init__(self, model_path: str = "checkpoints/yolo11n-face.pt", confidence: float = 0.5):
        super().__init__(confidence=confidence)
        self.model_path = model_path
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
            frame: Input image frame
            
        Returns:
            YOLO detection results or None
        """
        if self.model is None:
            return None
            
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            return results
        except Exception as e:
            print(f"❌ YOLO face detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert YOLO face detection results to dictionary format
        
        Args:
            results: YOLO detection results
            
        Returns:
            List of face landmark dictionaries or None
        """
        if not results or len(results) == 0:
            return None
        
        try:
            # YOLO face detection typically returns bounding boxes
            # We'll create simplified landmark data from the bounding box
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    
                    if len(boxes) > 0:
                        # Use the first detected face
                        box = boxes[0]
                        conf = confidences[0]
                        x1, y1, x2, y2 = box
                        
                        # Create simplified landmark data from bounding box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Create basic face landmarks (simplified)
                        landmark_data = []
                        
                        # Add some basic landmarks based on face bounding box
                        landmarks = [
                            {"x": center_x, "y": center_y, "name": "center"},  # Face center
                            {"x": x1, "y": y1, "name": "top_left"},  # Top left
                            {"x": x2, "y": y1, "name": "top_right"},  # Top right
                            {"x": x1, "y": y2, "name": "bottom_left"},  # Bottom left
                            {"x": x2, "y": y2, "name": "bottom_right"},  # Bottom right
                            {"x": center_x, "y": y1 + height * 0.4, "name": "nose"},  # Approximate nose
                            {"x": x1 + width * 0.3, "y": y1 + height * 0.3, "name": "left_eye"},  # Left eye
                            {"x": x2 - width * 0.3, "y": y1 + height * 0.3, "name": "right_eye"},  # Right eye
                            {"x": center_x, "y": y2 - height * 0.2, "name": "mouth"}  # Approximate mouth
                        ]
                        
                        for landmark in landmarks:
                            landmark_data.append({
                                "x": float(landmark["x"]) / frame.shape[1] if 'frame' in locals() else float(landmark["x"]) / 640,
                                "y": float(landmark["y"]) / frame.shape[0] if 'frame' in locals() else float(landmark["y"]) / 480,
                                "z": 0.0,  # YOLO doesn't provide depth
                                "confidence": float(conf)
                            })
                        
                        return landmark_data
                        
        except Exception as e:
            print(f"❌ Error converting YOLO face results: {e}")
            return None
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw face detection results on the frame
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            
        Returns:
            Frame with drawn detections
        """
        if not results or len(results) == 0:
            return frame
        
        try:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Draw confidence score
                        label = f"Face: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Draw simplified landmarks
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Draw basic landmarks
                        landmarks = [
                            (center_x, center_y, "center"),
                            (int(x1 + width * 0.3), int(y1 + height * 0.3), "L_eye"),
                            (int(x2 - width * 0.3), int(y1 + height * 0.3), "R_eye"),
                            (center_x, int(y1 + height * 0.4), "nose"),
                            (center_x, int(y2 - height * 0.2), "mouth")
                        ]
                        
                        for x, y, name in landmarks:
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                            cv2.putText(frame, name, (x + 3, y - 3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        
        except Exception as e:
            print(f"❌ Error drawing YOLO face results: {e}")
        
        return frame
    
    def get_face_bbox(self, results: Any) -> Optional[Dict]:
        """
        Get face bounding box from detection results
        
        Args:
            results: YOLO detection results
            
        Returns:
            Bounding box dictionary or None
        """
        if not results or len(results) == 0:
            return None
        
        try:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes.xyxy.cpu().numpy()[0]  # First face
                    conf = result.boxes.conf.cpu().numpy()[0]
                    x1, y1, x2, y2 = box
                    
                    return {
                        "x_min": float(x1),
                        "y_min": float(y1),
                        "x_max": float(x2),
                        "y_max": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "confidence": float(conf)
                    }
                    
        except Exception as e:
            print(f"❌ Error getting YOLO face bbox: {e}")
            return None
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "YOLO Face Detector",
            "type": "face",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "available": self.model is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # YOLO models don't need explicit cleanup
        pass