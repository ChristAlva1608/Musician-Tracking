from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List


class BaseEmotionDetector(ABC):
    """Abstract base class for emotion detection models"""
    
    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.model_type = "emotion"
        self.emotion_classes = []  # To be defined by subclasses
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect emotions in the frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Model-specific emotion detection results or None if no face/emotion detected
        """
        pass
    
    @abstractmethod
    def convert_to_dict(self, results: Dict) -> Optional[Dict]:
        """
        Convert emotion detection results to standardized dictionary format
        
        Args:
            results: Model-specific emotion detection results
            
        Returns:
            Standardized dictionary with keys:
            - emotions: Dict[str, float] - emotion scores
            - dominant_emotion: str - emotion with highest score
            - confidence: float - confidence of dominant emotion
            - face_detected: bool - whether face was detected
            - bbox: Optional[Dict] - face bounding box if available
        """
        pass
    
    @abstractmethod
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw emotion detection results on the frame
        
        Args:
            frame: Input image frame
            results: Emotion detection results
            
        Returns:
            Frame with drawn emotion information
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model metadata including name, type, confidence_threshold, emotion_classes, etc.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup resources and close model
        """
        pass
    
    def get_dominant_emotion(self, results: Dict) -> Optional[str]:
        """
        Get the dominant emotion from results
        
        Args:
            results: Emotion detection results
            
        Returns:
            Name of dominant emotion or None
        """
        converted = self.convert_to_dict(results)
        if not converted:
            return None
        return converted.get('dominant_emotion')
    
    def get_emotion_confidence(self, results: Dict) -> float:
        """
        Get the confidence of the dominant emotion
        
        Args:
            results: Emotion detection results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        converted = self.convert_to_dict(results)
        if not converted:
            return 0.0
        return converted.get('confidence', 0.0)
    
    def get_all_emotions(self, results: Dict) -> Dict[str, float]:
        """
        Get all emotion scores
        
        Args:
            results: Emotion detection results
            
        Returns:
            Dictionary mapping emotion names to scores
        """
        converted = self.convert_to_dict(results)
        if not converted:
            return {}
        return converted.get('emotions', {})
    
    def has_face_detected(self, results: Dict) -> bool:
        """
        Check if face was detected for emotion analysis
        
        Args:
            results: Emotion detection results
            
        Returns:
            True if face was detected, False otherwise
        """
        converted = self.convert_to_dict(results)
        if not converted:
            return False
        return converted.get('face_detected', False)
    
    def get_emotion_above_threshold(self, results: Dict, threshold: float = 0.5) -> List[str]:
        """
        Get emotions that are above a certain confidence threshold
        
        Args:
            results: Emotion detection results
            threshold: Minimum confidence threshold
            
        Returns:
            List of emotion names above threshold
        """
        emotions = self.get_all_emotions(results)
        return [emotion for emotion, score in emotions.items() if score >= threshold]
    
    def normalize_emotion_scores(self, results: Dict) -> Optional[Dict[str, float]]:
        """
        Normalize emotion scores to sum to 1.0
        
        Args:
            results: Emotion detection results
            
        Returns:
            Dictionary with normalized emotion scores or None
        """
        emotions = self.get_all_emotions(results)
        if not emotions:
            return None
        
        total = sum(emotions.values())
        if total == 0:
            return emotions
        
        return {emotion: score / total for emotion, score in emotions.items()}
    
    def get_emotion_vector(self, results: Dict) -> Optional[List[float]]:
        """
        Get emotion scores as a vector in the order of emotion_classes
        
        Args:
            results: Emotion detection results
            
        Returns:
            List of emotion scores in order of self.emotion_classes
        """
        emotions = self.get_all_emotions(results)
        if not emotions or not self.emotion_classes:
            return None
        
        return [emotions.get(emotion, 0.0) for emotion in self.emotion_classes]