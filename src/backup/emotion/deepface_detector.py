#!/usr/bin/env python3
"""
DeepFace Emotion Detector
Emotion detection using DeepFace library
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .base_emotion_detector import BaseEmotionDetector

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

class DeepFaceDetector(BaseEmotionDetector):
    """DeepFace emotion detector"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepFace detector
        
        Args:
            config: Configuration dictionary with DeepFace settings
        """
        super().__init__(config)
        self.model_name = "DeepFace"
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # DeepFace specific settings
        deepface_config = config.get('deepface', {})
        self.model_name_deepface = deepface_config.get('model_name', 'Facenet')
        self.detector_backend = deepface_config.get('detector_backend', 'retinaface')
        self.enforce_detection = deepface_config.get('enforce_detection', False)
        
        self.initialized = False
    
    def initialize_model(self) -> bool:
        """
        Initialize the DeepFace model
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace not available. Install with: pip install deepface")
            return False
        
        try:
            print(f"✅ DeepFace emotion detector initialized")
            print(f"  Model: {self.model_name_deepface}")
            print(f"  Detector backend: {self.detector_backend}")
            print(f"  Enforce detection: {self.enforce_detection}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize DeepFace: {e}")
            return False
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect emotions in a single frame using DeepFace
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of emotion detection results
        """
        if not self.initialized:
            if not self.initialize_model():
                return []
        
        if not self.validate_frame(frame):
            return []
        
        try:
            # Analyze frame with DeepFace
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            detections = []
            
            # Handle both single result and list of results
            if not isinstance(result, list):
                result = [result]
            
            for detection in result:
                if 'emotion' in detection and 'region' in detection:
                    # Get bounding box from region
                    region = detection['region']
                    bbox = (region['x'], region['y'], region['w'], region['h'])
                    
                    # Get emotion scores
                    emotion_scores = detection['emotion']
                    
                    # Normalize emotion scores to match our standard format
                    normalized_scores = {}
                    for emotion in self.emotion_classes:
                        if emotion in emotion_scores:
                            normalized_scores[emotion] = emotion_scores[emotion] / 100.0
                        else:
                            normalized_scores[emotion] = 0.0
                    
                    # Get dominant emotion
                    dominant_emotion, confidence = self.get_dominant_emotion(normalized_scores)
                    
                    detections.append({
                        'bbox': bbox,
                        'emotions': normalized_scores,
                        'dominant_emotion': dominant_emotion,
                        'confidence': confidence
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in DeepFace emotion detection: {e}")
            return []
    
    def cleanup(self):
        """
        Cleanup DeepFace resources
        """
        # DeepFace doesn't require explicit cleanup
        self.initialized = False