import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .base_pose_detector import BasePoseDetector


class MediaPipePoseDetector(BasePoseDetector):
    """MediaPipe pose detection model"""
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_landmarks = mp.solutions.pose.PoseLandmark
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect pose in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            MediaPipe pose detection results or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_frame)
        return results
    
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert MediaPipe pose landmarks to dictionary format

        Args:
            results: MediaPipe pose detection results

        Returns:
            List of pose landmark dictionaries or None
        """
        if not results or not results.pose_world_landmarks:
            return None

        landmark_data = []
        for landmark in results.pose_world_landmarks.landmark:
            landmark_data.append({
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "confidence": float(landmark.visibility)
            })

        return landmark_data
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw pose landmarks on the frame
        
        Args:
            frame: Input image frame
            results: MediaPipe pose detection results
            
        Returns:
            Frame with drawn landmarks
        """
        if results and results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.pose_connections
            )
        return frame
    
    def get_landmark_by_name(self, results: Any, landmark_name: str) -> Optional[Dict]:
        """
        Get specific landmark by name

        Args:
            results: MediaPipe pose detection results
            landmark_name: Name of the landmark (e.g., 'LEFT_WRIST', 'RIGHT_SHOULDER')

        Returns:
            Landmark dictionary or None
        """
        if not results or not results.pose_world_landmarks:
            return None

        try:
            landmark_index = getattr(self.pose_landmarks, landmark_name)
            landmark = results.pose_world_landmarks.landmark[landmark_index]
            return {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "confidence": float(landmark.visibility)
            }
        except AttributeError:
            print(f"❌ Unknown landmark name: {landmark_name}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe Pose",
            "type": "pose",
            "confidence_threshold": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence,
            "num_landmarks": 33  # MediaPipe pose has 33 landmarks
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.close()