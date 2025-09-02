#!/usr/bin/env python3
"""
GhostFaceNet Emotion Detector
Emotion detection using GhostFaceNetsV2 model
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from typing import Dict, List, Tuple, Optional, Any
from .base_emotion_detector import BaseEmotionDetector

try:
    import torch
    import torch.nn.functional as F
    from ellzaf_ml.models import GhostFaceNetsV2
    from ellzaf_ml.tools.load_pretrained import load_pretrained
    GHOSTFACENET_AVAILABLE = True
except ImportError:
    GHOSTFACENET_AVAILABLE = False

class GhostFaceNetDetector(BaseEmotionDetector):
    """GhostFaceNet emotion detector"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GhostFaceNet detector
        
        Args:
            config: Configuration dictionary with GhostFaceNet settings
        """
        super().__init__(config)
        self.model_name = "GhostFaceNet"
        self.emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # GhostFaceNet specific settings
        ghostface_config = config.get('ghostfacenet', {})
        self.model_version = ghostface_config.get('model_version', 'v2')
        self.batch_size = ghostface_config.get('batch_size', 32)
        self.image_size = 224
        
        # Device settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.face_detection = None
        self.initialized = False
        
        # Find checkpoint path
        self.checkpoint_path = self._find_checkpoint()
    
    def _find_checkpoint(self) -> Optional[str]:
        """
        Find GhostFaceNet checkpoint file
        
        Returns:
            Path to checkpoint if found, None otherwise
        """
        # Look for checkpoint files in common locations
        checkpoint_patterns = [
            'checkpoints/*.pth',
            'checkpoints/ghostface*.pth',
            'models/*.pth',
            '*.pth'
        ]
        
        from glob import glob
        for pattern in checkpoint_patterns:
            files = glob(pattern)
            if files:
                return files[0]  # Return first found checkpoint
        
        return None
    
    def initialize_model(self) -> bool:
        """
        Initialize the GhostFaceNet model
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not GHOSTFACENET_AVAILABLE:
            print("❌ GhostFaceNet dependencies not available")
            print("Required: torch, ellzaf_ml")
            return False
        
        if self.checkpoint_path is None:
            print("❌ No GhostFaceNet checkpoint found")
            return False
        
        try:
            # Create GhostFaceNetsV2 model
            self.model = GhostFaceNetsV2(
                image_size=self.image_size,
                num_classes=len(self.emotion_classes),
                width=1.0,
                dropout=0.2
            )
            
            # Load pretrained weights
            print(f"Loading GhostFaceNet weights from {self.checkpoint_path}")
            try:
                load_pretrained(self.model, self.checkpoint_path, not_vit=True)
                print("✅ Weights loaded successfully")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
                try:
                    load_pretrained(self.model, self.checkpoint_path, not_vit=False)
                    print("✅ Weights loaded with alternative method")
                except Exception as e2:
                    print(f"❌ Alternative loading also failed: {e2}")
                    return False
            
            # Move model to device and set evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize face detection
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            print(f"✅ GhostFaceNet emotion detector initialized")
            print(f"  Device: {self.device}")
            print(f"  Image size: {self.image_size}")
            print(f"  Model version: {self.model_version}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize GhostFaceNet: {e}")
            return False
    
    def preprocess_face_image(self, face_roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for GhostFaceNet
        
        Args:
            face_roi: Face region of interest as numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(face_resized).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_roi: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from face image
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Tuple of (emotion_label, confidence, all_predictions)
        """
        try:
            # Preprocess image
            face_tensor = self.preprocess_face_image(face_roi)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                emotion_label = self.emotion_classes[predicted_idx.item()]
                confidence_score = confidence.item()
                all_predictions = probabilities[0].cpu().numpy()
                
                return emotion_label, confidence_score, all_predictions
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Neutral", 0.0, np.zeros(len(self.emotion_classes))
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect emotions in a single frame using GhostFaceNet
        
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
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            detections = []
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Predict emotion
                        emotion, confidence, all_predictions = self.predict_emotion(face_roi)
                        
                        # Convert predictions to emotion scores dictionary
                        emotion_scores = self.get_emotion_scores_dict(all_predictions)
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'emotions': emotion_scores,
                            'dominant_emotion': emotion.lower(),  # Convert to lowercase for consistency
                            'confidence': confidence
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in GhostFaceNet emotion detection: {e}")
            return []
    
    def cleanup(self):
        """
        Cleanup GhostFaceNet resources
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.face_detection is not None:
            self.face_detection.close()
            self.face_detection = None
        
        # Clear CUDA cache if using GPU
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.initialized = False