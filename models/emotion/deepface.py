import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion.deepface_detector import DeepFaceDetector as BaseDeepFaceDetector
from .base_emotion_detector import BaseEmotionDetector
import cv2
import numpy as np
from typing import Optional, Dict, Any, List


class DeepFaceEmotionDetector(BaseEmotionDetector):
    """Wrapper for DeepFace emotion detection model"""
    
    def __init__(self, model_name: str = "Facenet", 
                 detector_backend: str = "retinaface",
                 enforce_detection: bool = False):
        super().__init__()
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.emotion_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        try:
            self.detector = BaseDeepFaceDetector(
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection
            )
            print(f"✅ DeepFace emotion detector loaded: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load DeepFace emotion detector: {e}")
            self.detector = None
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect emotions in the frame using DeepFace
        
        Args:
            frame: Input image frame
            
        Returns:
            Emotion detection results or None
        """
        if self.detector is None:
            return None
            
        try:
            results = self.detector.detect_emotion(frame)
            return results
        except Exception as e:
            print(f"❌ DeepFace emotion detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Dict) -> Optional[Dict]:
        """
        Convert DeepFace emotion results to standardized dictionary format
        
        Args:
            results: DeepFace emotion detection results
            
        Returns:
            Standardized emotion dictionary or None
        """
        if not results:
            return None
        
        try:
            # DeepFace results are already in dictionary format
            return {
                "emotions": results.get("emotions", {}),
                "dominant_emotion": results.get("dominant_emotion", "neutral"),
                "confidence": results.get("confidence", 0.0),
                "face_detected": results.get("face_detected", False),
                "bbox": results.get("bbox", None)
            }
        except Exception as e:
            print(f"❌ Error converting DeepFace results: {e}")
            return None
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw emotion detection results on the frame
        
        Args:
            frame: Input image frame
            results: Emotion detection results
            
        Returns:
            Frame with drawn results
        """
        if not results or not results.get("face_detected", False):
            return frame
        
        try:
            # Draw bounding box if available
            bbox = results.get("bbox")
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw dominant emotion
            dominant_emotion = results.get("dominant_emotion", "unknown")
            confidence = results.get("confidence", 0.0)
            
            label = f"{dominant_emotion}: {confidence:.2f}"
            y_pos = bbox[1] - 10 if bbox else 30
            x_pos = bbox[0] if bbox else 10
            
            cv2.putText(frame, label, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw all emotion scores
            emotions = results.get("emotions", {})
            for i, (emotion, score) in enumerate(emotions.items()):
                y_pos = (bbox[1] + bbox[3] + 20 + i * 20) if bbox else (60 + i * 20)
                x_pos = bbox[0] if bbox else 10
                
                emotion_label = f"{emotion}: {score:.3f}"
                cv2.putText(frame, emotion_label, (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                           
        except Exception as e:
            print(f"❌ Error drawing DeepFace results: {e}")
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "DeepFace Emotion Detector",
            "type": "emotion",
            "model_name": self.model_name,
            "detector_backend": self.detector_backend,
            "enforce_detection": self.enforce_detection,
            "available": self.detector is not None,
            "emotion_classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # DeepFace doesn't need explicit cleanup
        pass