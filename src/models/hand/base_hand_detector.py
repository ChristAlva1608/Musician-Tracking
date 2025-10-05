from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class BaseHandDetector(ABC):
    """Abstract base class for hand detection models"""
    
    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.model_type = "hand"
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect hands in the frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Model-specific detection results or None if no hands detected
        """
        pass
    
    @abstractmethod
    def convert_to_dict(self, results: Any) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Convert hand detection results to standardized dictionary format
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
            Each hand landmark list contains dictionaries with keys: x, y, z, confidence
        """
        pass
    
    @abstractmethod
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand landmarks on the frame
        
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
            Dictionary containing model metadata including name, type, confidence_threshold, etc.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup resources and close model
        """
        pass
    
    def get_hand_landmarks_by_side(self, results: Any, side: str) -> Optional[List[Dict]]:
        """
        Get landmarks for a specific hand side
        
        Args:
            results: Model-specific detection results
            side: "left" or "right"
            
        Returns:
            List of landmark dictionaries or None
        """
        left_hand, right_hand = self.convert_to_dict(results)
        
        if side.lower() == "left":
            return left_hand
        elif side.lower() == "right":
            return right_hand
        else:
            raise ValueError("Side must be 'left' or 'right'")
    
    def has_hands_detected(self, results: Any) -> bool:
        """
        Check if any hands are detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            True if hands are detected, False otherwise
        """
        left_hand, right_hand = self.convert_to_dict(results)
        return left_hand is not None or right_hand is not None
    
    def get_hand_count(self, results: Any) -> int:
        """
        Get the number of hands detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Number of hands detected (0, 1, or 2)
        """
        left_hand, right_hand = self.convert_to_dict(results)
        count = 0
        if left_hand is not None:
            count += 1
        if right_hand is not None:
            count += 1
        return count