import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .base_hand_detector import BaseHandDetector


class MediaPipeHandDetector(BaseHandDetector):
    """MediaPipe hand detection model"""
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.hand_landmarks = mp.solutions.hands.HandLandmark
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect hands in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            MediaPipe hand detection results or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_frame)
        return results
    
    def convert_to_dict(self, results: Any) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Convert MediaPipe hand landmarks to dictionary format
        
        Args:
            results: MediaPipe hand detection results
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        left_hand = None
        right_hand = None
        
        if results and results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append({
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "confidence": 1.0  # MediaPipe doesn't provide per-landmark confidence for hands
                    })
                
                # For simplicity, assume first hand is left, second is right
                # In production, you'd use handedness detection from results.multi_handedness
                if i == 0:
                    left_hand = hand_data
                elif i == 1:
                    right_hand = hand_data
        
        return left_hand, right_hand
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand landmarks on the frame
        
        Args:
            frame: Input image frame
            results: MediaPipe hand detection results
            
        Returns:
            Frame with drawn landmarks
        """
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe Hands",
            "type": "hand",
            "confidence_threshold": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.close()