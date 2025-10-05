from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Dict, Any


class BasePoseDetector(ABC):
    """Abstract base class for pose detection models"""
    
    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.model_type = "pose"
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect pose in the frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Model-specific detection results or None if no pose detected
        """
        pass
    
    @abstractmethod
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert pose detection results to standardized dictionary format
        
        Args:
            results: Model-specific detection results
            
        Returns:
            List of pose landmark dictionaries with keys: x, y, z, confidence
            Returns None if no pose detected
        """
        pass
    
    @abstractmethod
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw pose landmarks on the frame
        
        Args:
            frame: Input image frame
            results: Model-specific detection results
            
        Returns:
            Frame with drawn landmarks
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model metadata including name, type, confidence_threshold, num_landmarks, etc.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup resources and close model
        """
        pass
    
    def get_landmark_by_name(self, results: Any, landmark_name: str) -> Optional[Dict]:
        """
        Get specific landmark by name (to be implemented by subclasses if supported)
        
        Args:
            results: Model-specific detection results
            landmark_name: Name of the landmark
            
        Returns:
            Landmark dictionary or None
        """
        # Default implementation returns None, subclasses should override
        return None
    
    def has_pose_detected(self, results: Any) -> bool:
        """
        Check if pose is detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            True if pose is detected, False otherwise
        """
        landmarks = self.convert_to_dict(results)
        return landmarks is not None and len(landmarks) > 0
    
    def get_landmark_count(self, results: Any) -> int:
        """
        Get the number of landmarks detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Number of landmarks detected
        """
        landmarks = self.convert_to_dict(results)
        return len(landmarks) if landmarks else 0
    
    def get_body_parts(self, results: Any) -> Dict[str, Optional[Dict]]:
        """
        Get commonly used body parts (generic implementation)
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Dictionary with body part names and their landmarks
        """
        # Generic body part names that might be supported
        body_parts = {
            'left_wrist': None,
            'right_wrist': None,
            'left_elbow': None,
            'right_elbow': None,
            'left_shoulder': None,
            'right_shoulder': None,
            'nose': None
        }
        
        # Try to get landmarks by name (will be None if not implemented)
        for part in body_parts.keys():
            body_parts[part] = self.get_landmark_by_name(results, part)
        
        return body_parts