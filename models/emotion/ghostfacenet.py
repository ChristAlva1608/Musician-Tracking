from .base_emotion_detector import BaseEmotionDetector
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class GhostFaceNetEmotionDetector(BaseEmotionDetector):
    """GhostFaceNet emotion detection model (simplified version)"""
    
    def __init__(self, model_version: str = "v2", batch_size: int = 32):
        super().__init__()
        self.model_version = model_version
        self.batch_size = batch_size
        self.emotion_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.available = TORCH_AVAILABLE and MEDIAPIPE_AVAILABLE
        
        if not self.available:
            print("❌ GhostFaceNet dependencies not available (requires torch and mediapipe)")
        else:
            print(f"✅ GhostFaceNet emotion detector initialized: {model_version}")
            print("⚠️ Note: This is a simplified implementation without the actual GhostFaceNet model")
        
        self.face_detection = None
        if MEDIAPIPE_AVAILABLE:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect emotions in the frame using GhostFaceNet (simplified)
        
        Args:
            frame: Input image frame
            
        Returns:
            Emotion detection results or None
        """
        if not self.available or self.face_detection is None:
            return None
            
        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                # Use first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                # For now, return dummy emotion scores since we don't have the actual model
                # In a real implementation, this would process the face through GhostFaceNet
                dummy_emotions = {
                    'neutral': 0.7,
                    'happy': 0.15,
                    'sad': 0.1,
                    'angry': 0.02,
                    'surprise': 0.02,
                    'fear': 0.01,
                    'disgust': 0.0
                }
                
                return {
                    'emotions': dummy_emotions,
                    'dominant_emotion': 'neutral',
                    'confidence': 0.7,
                    'face_detected': True,
                    'bbox': (x, y, w, h)
                }
            
            return None
            
        except Exception as e:
            print(f"❌ GhostFaceNet emotion detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Dict) -> Optional[Dict]:
        """
        Convert GhostFaceNet emotion results to standardized dictionary format
        
        Args:
            results: GhostFaceNet emotion detection results
            
        Returns:
            Standardized emotion dictionary or None
        """
        if not results:
            return None
        
        try:
            # GhostFaceNet results should already be in dictionary format
            return {
                "emotions": results.get("emotions", {}),
                "dominant_emotion": results.get("dominant_emotion", "neutral"),
                "confidence": results.get("confidence", 0.0),
                "face_detected": results.get("face_detected", False),
                "bbox": results.get("bbox", None)
            }
        except Exception as e:
            print(f"❌ Error converting GhostFaceNet results: {e}")
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw dominant emotion
            dominant_emotion = results.get("dominant_emotion", "unknown")
            confidence = results.get("confidence", 0.0)
            
            label = f"GFN-{dominant_emotion}: {confidence:.2f}"
            y_pos = bbox[1] - 10 if bbox else 30
            x_pos = bbox[0] if bbox else 10
            
            cv2.putText(frame, label, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw all emotion scores
            emotions = results.get("emotions", {})
            for i, (emotion, score) in enumerate(emotions.items()):
                y_pos = (bbox[1] + bbox[3] + 20 + i * 20) if bbox else (60 + i * 20)
                x_pos = bbox[0] if bbox else 10
                
                emotion_label = f"{emotion}: {score:.3f}"
                cv2.putText(frame, emotion_label, (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                           
        except Exception as e:
            print(f"❌ Error drawing GhostFaceNet results: {e}")
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "GhostFaceNet Emotion Detector",
            "type": "emotion",
            "model_version": self.model_version,
            "batch_size": self.batch_size,
            "available": self.available,
            "emotion_classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # GhostFaceNet doesn't need explicit cleanup
        pass