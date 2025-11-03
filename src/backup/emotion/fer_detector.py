#!/usr/bin/env python3
"""
FER Emotion Detector
Emotion detection using FER (Facial Emotion Recognition) library with MTCNN
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .base_emotion_detector import BaseEmotionDetector

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

class FERDetector(BaseEmotionDetector):
    """FER emotion detector with MTCNN"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FER detector
        
        Args:
            config: Configuration dictionary with FER settings
        """
        super().__init__(config)
        self.model_name = "FER"
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # FER specific settings
        fer_config = config.get('fer', {})
        self.use_mtcnn = fer_config.get('use_mtcnn', True)
        self.min_face_size = fer_config.get('min_face_size', 40)
        
        self.detector = None
        self.initialized = False
    
    def initialize_model(self) -> bool:
        """
        Initialize the FER model
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not FER_AVAILABLE:
            print("❌ FER not available. Install with: pip install fer")
            return False
        
        try:
            # Initialize FER with MTCNN detector
            self.detector = FER(mtcnn=self.use_mtcnn)
            
            print(f"✅ FER emotion detector initialized")
            print(f"  MTCNN enabled: {self.use_mtcnn}")
            print(f"  Min face size: {self.min_face_size}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize FER: {e}")
            return False
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect emotions in a single frame using FER
        
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
            # Detect emotions using FER
            emotions = self.detector.detect_emotions(frame)
            
            if not emotions:
                return []
            
            detections = []
            
            # Convert FER format to our standard format
            for emotion_data in emotions:
                # Get bounding box
                bbox = emotion_data['box']
                x, y, w, h = bbox
                
                # Filter out very small faces
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                
                # Get emotions dictionary
                emotion_scores = emotion_data['emotions']
                
                # Normalize scores (FER already provides values 0-1)
                normalized_scores = {}
                for emotion in self.emotion_classes:
                    if emotion in emotion_scores:
                        normalized_scores[emotion] = emotion_scores[emotion]
                    else:
                        normalized_scores[emotion] = 0.0
                
                # Get dominant emotion
                dominant_emotion, confidence = self.get_dominant_emotion(normalized_scores)
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'emotions': normalized_scores,
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence
                })
            
            return detections
            
        except Exception as e:
            print(f"Error in FER emotion detection: {e}")
            return []
    
    def cleanup(self):
        """
        Cleanup FER resources
        """
        # FER doesn't require explicit cleanup, but we can clear the detector
        self.detector = None
        self.initialized = False