#!/usr/bin/env python3
"""
Emotion Detection Models Package
"""

from .base_emotion_detector import BaseEmotionDetector
from .deepface import DeepFaceEmotionDetector
from .ghostfacenet import GhostFaceNetEmotionDetector

__all__ = [
    'BaseEmotionDetector',
    'DeepFaceEmotionDetector', 
    'GhostFaceNetEmotionDetector'
]