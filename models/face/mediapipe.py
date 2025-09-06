import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .base_face_detector import BaseFaceDetector


class MediaPipeFaceDetector(BaseFaceDetector):
    """MediaPipe face detection model using FaceLandmarker"""
    
    def __init__(self, model_path: str = "checkpoints/face_landmarker.task", 
                 min_detection_confidence: float = 0.5,
                 num_faces: int = 1):
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.num_faces = num_faces
        self.model_path = model_path
        self.model = None
        
        try:
            # Use new FaceLandmarker API
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=num_faces,
                min_face_detection_confidence=min_detection_confidence
            )
            self.model = vision.FaceLandmarker.create_from_options(options)
            print(f"✅ MediaPipe FaceLandmarker loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load MediaPipe FaceLandmarker: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect face landmarks in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            MediaPipe face detection results or None
        """
        if self.model is None:
            return None
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Detect face landmarks
            results = self.model.detect(mp_image)
            return results
        except Exception as e:
            print(f"❌ MediaPipe face detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert MediaPipe face landmarks to dictionary format
        
        Args:
            results: MediaPipe face detection results
            
        Returns:
            List of face landmark dictionaries or None
        """
        if not results or not results.face_landmarks:
            return None
        
        try:
            # Get landmarks from the first detected face
            face_landmarks = results.face_landmarks[0]
            landmark_data = []
            
            for landmark in face_landmarks:
                landmark_data.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "confidence": 1.0  # MediaPipe FaceLandmarker doesn't provide per-landmark confidence
                })
            
            return landmark_data
            
        except Exception as e:
            print(f"❌ Error converting MediaPipe face results: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw face landmarks on the frame
        
        Args:
            frame: Input image frame
            results: MediaPipe face detection results
            
        Returns:
            Frame with drawn landmarks
        """
        if not results or not results.face_landmarks:
            return frame
        
        try:
            height, width = frame.shape[:2]
            
            for face_landmarks in results.face_landmarks:
                # Draw face landmarks
                for landmark in face_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
        except Exception as e:
            print(f"❌ Error drawing MediaPipe face results: {e}")
        
        return frame
    
    def get_face_bbox(self, results: Any) -> Optional[Dict]:
        """
        Get face bounding box from landmarks
        
        Args:
            results: MediaPipe face detection results
            
        Returns:
            Bounding box dictionary or None
        """
        if not results or not results.face_landmarks:
            return None
        
        try:
            face_landmarks = results.face_landmarks[0]
            
            # Calculate bounding box from landmarks
            x_coords = [landmark.x for landmark in face_landmarks]
            y_coords = [landmark.y for landmark in face_landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            return {
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "width": float(x_max - x_min),
                "height": float(y_max - y_min)
            }
            
        except Exception as e:
            print(f"❌ Error calculating face bbox: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe FaceLandmarker",
            "type": "face",
            "model_path": self.model_path,
            "confidence_threshold": self.min_detection_confidence,
            "num_faces": self.num_faces,
            "num_landmarks": 468,  # MediaPipe face mesh has 468 landmarks
            "available": self.model is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # MediaPipe FaceLandmarker doesn't need explicit cleanup
        pass