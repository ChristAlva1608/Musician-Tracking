from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .base_pose_detector import BasePoseDetector


class YOLOPoseDetector(BasePoseDetector):
    """YOLO pose detection model"""
    
    def __init__(self, model_path: str = "src/checkpoints/yolo11n-pose.pt", confidence: float = 0.5):
        super().__init__(confidence=confidence)
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO pose model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO pose model: {e}")
            self.model = None
            
        # YOLO pose keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect pose in the frame using YOLO
        
        Args:
            frame: Input image frame
            
        Returns:
            YOLO detection results or None
        """
        if self.model is None:
            return None
            
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            return results
        except Exception as e:
            print(f"❌ YOLO pose detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert YOLO pose detection results to dictionary format
        
        Args:
            results: YOLO detection results
            
        Returns:
            List of pose landmark dictionaries or None
        """
        if not results or len(results) == 0 or results[0].keypoints is None:
            return None
        
        try:
            # Get keypoints from the first detected person
            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) == 0:
                return None
                
            # Get the first person's keypoints
            person_keypoints = keypoints[0]
            landmark_data = []
            
            for i, (x, y) in enumerate(person_keypoints):
                # YOLO provides x,y coordinates, we normalize and add z=0
                landmark_data.append({
                    "x": float(x) if x > 0 else 0.0,
                    "y": float(y) if y > 0 else 0.0,
                    "z": 0.0,  # YOLO doesn't provide z-coordinate
                    "confidence": 1.0 if x > 0 and y > 0 else 0.0  # Valid if both x,y > 0
                })
            
            return landmark_data
            
        except Exception as e:
            print(f"❌ Error converting YOLO pose results: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw pose landmarks on the frame
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            
        Returns:
            Frame with drawn landmarks
        """
        if not results or len(results) == 0 or results[0].keypoints is None:
            return frame
        
        try:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) == 0:
                return frame
                
            # Draw keypoints for the first detected person
            person_keypoints = keypoints[0]
            
            # Draw keypoints
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Draw skeleton connections (simplified)
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(person_keypoints) and 
                    end_idx < len(person_keypoints)):
                    start_point = person_keypoints[start_idx]
                    end_point = person_keypoints[end_idx]
                    
                    if (start_point[0] > 0 and start_point[1] > 0 and 
                        end_point[0] > 0 and end_point[1] > 0):
                        cv2.line(frame, 
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                (255, 0, 0), 2)
                        
        except Exception as e:
            print(f"❌ Error drawing YOLO pose results: {e}")
        
        return frame
    
    def get_landmark_by_name(self, results: Any, landmark_name: str) -> Optional[Dict]:
        """
        Get specific landmark by name
        
        Args:
            results: YOLO pose detection results
            landmark_name: Name of the landmark (e.g., 'left_wrist', 'right_shoulder')
            
        Returns:
            Landmark dictionary or None
        """
        if not results or len(results) == 0 or results[0].keypoints is None:
            return None
        
        try:
            landmark_name = landmark_name.lower()
            if landmark_name not in self.keypoint_names:
                print(f"❌ Unknown landmark name: {landmark_name}")
                return None
            
            landmark_index = self.keypoint_names.index(landmark_name)
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            if len(keypoints) == 0 or landmark_index >= len(keypoints[0]):
                return None
            
            person_keypoints = keypoints[0]
            x, y = person_keypoints[landmark_index]
            
            return {
                "x": float(x) if x > 0 else 0.0,
                "y": float(y) if y > 0 else 0.0,
                "z": 0.0,
                "confidence": 1.0 if x > 0 and y > 0 else 0.0
            }
            
        except Exception as e:
            print(f"❌ Error getting YOLO landmark: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "YOLO Pose Detector",
            "type": "pose",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "num_landmarks": len(self.keypoint_names),
            "available": self.model is not None,
            "keypoint_names": self.keypoint_names
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # YOLO models don't need explicit cleanup
        pass