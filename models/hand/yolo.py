from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .base_hand_detector import BaseHandDetector


class YOLOHandDetector(BaseHandDetector):
    """YOLO hand detection model"""
    
    def __init__(self, model_path: str = "checkpoints/yolo11n-hand.pt", confidence: float = 0.5):
        super().__init__(confidence=confidence)
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO hand model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO hand model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect hands in the frame using YOLO
        
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
            print(f"❌ YOLO hand detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Any) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Convert YOLO hand detection results to dictionary format
        
        Args:
            results: YOLO detection results
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        if not results or len(results) == 0:
            return None, None
        
        left_hand = None
        right_hand = None
        
        try:
            # YOLO hand detection typically returns bounding boxes, not landmarks
            # This is a simplified conversion - you might need to adapt based on your specific YOLO hand model
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Create simplified hand data with bounding box center
                        hand_data = [{
                            "x": float(center_x / frame.shape[1]) if 'frame' in locals() else float(center_x / 640),
                            "y": float(center_y / frame.shape[0]) if 'frame' in locals() else float(center_y / 480),
                            "z": 0.0,  # YOLO doesn't provide depth
                            "confidence": float(conf)
                        }]
                        
                        # Assign to left or right hand based on detection order
                        if i == 0:
                            left_hand = hand_data
                        elif i == 1:
                            right_hand = hand_data
                        
        except Exception as e:
            print(f"❌ Error converting YOLO hand results: {e}")
            return None, None
        
        return left_hand, right_hand
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand detection results on the frame
        
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw confidence score
                        label = f"Hand: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"❌ Error drawing YOLO hand results: {e}")
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "YOLO Hand Detector",
            "type": "hand",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "available": self.model is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # YOLO models don't need explicit cleanup
        pass