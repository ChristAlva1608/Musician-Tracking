#!/usr/bin/env python3
"""
Base Emotion Detector Class
Abstract base class for all emotion detection models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class BaseEmotionDetector(ABC):
    """Abstract base class for emotion detection models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize emotion detector
        
        Args:
            config: Configuration dictionary containing model-specific settings
        """
        self.config = config
        self.emotion_classes = []
        self.model_name = ""
        
    @abstractmethod
    def initialize_model(self) -> bool:
        """
        Initialize the emotion detection model
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def detect_emotions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect emotions in a single frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of emotion detection results, each containing:
            - bbox: Tuple of (x, y, w, h) for face bounding box
            - emotions: Dict of emotion_name -> confidence_score
            - dominant_emotion: String name of most confident emotion
            - confidence: Float confidence score for dominant emotion
        """
        pass
    
    def get_emotion_scores_dict(self, emotions_array: np.ndarray) -> Dict[str, float]:
        """
        Convert emotion scores array to dictionary
        
        Args:
            emotions_array: Numpy array of emotion scores
            
        Returns:
            Dictionary mapping emotion names to scores
        """
        if len(emotions_array) != len(self.emotion_classes):
            raise ValueError(f"Emotions array length {len(emotions_array)} doesn't match classes length {len(self.emotion_classes)}")
            
        return {emotion: float(score) for emotion, score in zip(self.emotion_classes, emotions_array)}
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Get dominant emotion and its confidence
        
        Args:
            emotion_scores: Dictionary of emotion name -> confidence score
            
        Returns:
            Tuple of (dominant_emotion_name, confidence_score)
        """
        if not emotion_scores:
            return "neutral", 0.0
            
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return dominant_emotion, confidence
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate input frame
        
        Args:
            frame: Input image frame
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False
            
        if len(frame.shape) != 3:
            return False
            
        if frame.shape[2] != 3:
            return False
            
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'emotion_classes': self.emotion_classes,
            'config': self.config
        }
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup model resources
        """
        pass