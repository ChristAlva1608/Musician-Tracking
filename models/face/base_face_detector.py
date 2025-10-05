from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Dict, Any


class BaseFaceDetector(ABC):
    """Abstract base class for face detection models"""
    
    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.model_type = "face"
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect face landmarks in the frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Model-specific detection results or None if no face detected
        """
        pass
    
    @abstractmethod
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert face detection results to standardized dictionary format
        
        Args:
            results: Model-specific detection results
            
        Returns:
            List of face landmark dictionaries with keys: x, y, z, confidence
            Returns None if no face detected
        """
        pass
    
    @abstractmethod
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw face landmarks on the frame
        
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
    
    def get_face_bbox(self, results: Any) -> Optional[Dict]:
        """
        Get face bounding box (to be implemented by subclasses if supported)
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Bounding box dictionary with keys: x_min, y_min, x_max, y_max, width, height
            Returns None if not supported or no face detected
        """
        # Default implementation returns None, subclasses should override
        return None
    
    def has_face_detected(self, results: Any) -> bool:
        """
        Check if face is detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            True if face is detected, False otherwise
        """
        landmarks = self.convert_to_dict(results)
        return landmarks is not None and len(landmarks) > 0
    
    def get_landmark_count(self, results: Any) -> int:
        """
        Get the number of face landmarks detected
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Number of landmarks detected
        """
        landmarks = self.convert_to_dict(results)
        return len(landmarks) if landmarks else 0
    
    def get_face_center(self, results: Any) -> Optional[Dict]:
        """
        Get the center point of the detected face
        
        Args:
            results: Model-specific detection results
            
        Returns:
            Dictionary with x, y coordinates of face center or None
        """
        landmarks = self.convert_to_dict(results)
        if not landmarks:
            return None
        
        # Calculate center from landmarks
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return {
            'x': center_x,
            'y': center_y,
            'z': 0.0,
            'confidence': 1.0
        }
    
    def extract_face_region(self, frame: np.ndarray, results: Any, padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face region from frame using detection results
        
        Args:
            frame: Input image frame
            results: Model-specific detection results
            padding: Additional padding around face (as fraction of face size)
            
        Returns:
            Cropped face image or None if no face detected
        """
        bbox = self.get_face_bbox(results)
        if not bbox:
            return None
        
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates if needed
        if bbox['x_min'] <= 1.0 and bbox['y_min'] <= 1.0:
            x1 = int(bbox['x_min'] * width)
            y1 = int(bbox['y_min'] * height)
            x2 = int(bbox['x_max'] * width)
            y2 = int(bbox['y_max'] * height)
        else:
            x1, y1, x2, y2 = int(bbox['x_min']), int(bbox['y_min']), int(bbox['x_max']), int(bbox['y_max'])
        
        # Add padding
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * padding), int(h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
        
        return frame[y1:y2, x1:x2]