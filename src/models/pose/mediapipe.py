import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from typing import Optional, List, Dict, Any
from .base_pose_detector import BasePoseDetector


class MediaPipePoseDetector(BasePoseDetector):
    """MediaPipe pose detection model using new tasks API with multi-person support"""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 num_poses: int = 2, model_path: str = None):
        """
        Initialize MediaPipe Pose detector with new tasks API

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            num_poses: Maximum number of people to detect (default: 2)
            model_path: Path to .task model file (default: pose_landmarker_heavy.task)
        """
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_poses = num_poses

        # Default model path
        if model_path is None:
            # Try to find model in checkpoints directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            model_path = os.path.join(project_root, "src", "checkpoints", "pose_landmarker_heavy.task")

            # Fallback to lite model if heavy not found
            if not os.path.exists(model_path):
                model_path = os.path.join(project_root, "src", "checkpoints", "pose_landmarker.task")

        self.model_path = model_path

        # Initialize model with new tasks API
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.model = vision.PoseLandmarker.create_from_options(options)
            print(f"✅ MediaPipe Pose (tasks API) loaded: {num_poses} people, model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Failed to load MediaPipe Pose model: {e}")
            print(f"   Download model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models")
            self.model = None

        # Drawing utilities (using legacy for compatibility)
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_landmarks_class = mp.solutions.pose.PoseLandmark
        # Use pose connections from the legacy API (compatible with drawing_utils)
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS

        # Frame counter for video processing
        self.frame_timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect pose in the frame (supports multiple people)

        Args:
            frame: Input image frame

        Returns:
            MediaPipe pose detection results or None
        """
        if self.model is None:
            print("❌ Pose model is None - model did not load properly!")
            return None

        try:
            # Convert to RGB and create MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose in the image
            results = self.model.detect(mp_image)

            return results
        except Exception as e:
            print(f"❌ Pose detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert MediaPipe pose landmarks to dictionary format
        Returns first person's landmarks for backward compatibility

        Args:
            results: MediaPipe pose detection results

        Returns:
            List of pose landmark dictionaries for first person, or None
        """
        if not results or not results.pose_world_landmarks:
            return None

        # Get first person's landmarks for backward compatibility
        if len(results.pose_world_landmarks) > 0:
            person_landmarks = results.pose_world_landmarks[0]
            landmark_data = []
            for landmark in person_landmarks:
                landmark_data.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "confidence": float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
                })
            return landmark_data

        return None

    def convert_to_dict_multi_person(self, results: Any) -> Optional[List[List[Dict]]]:
        """
        Convert MediaPipe pose landmarks to dictionary format for all detected people

        Args:
            results: MediaPipe pose detection results

        Returns:
            List of people, each containing list of landmark dictionaries
            Format: [[person1_landmarks], [person2_landmarks], ...]
        """
        if not results or not results.pose_world_landmarks:
            return None

        all_people_landmarks = []

        # Iterate through each detected person
        for person_landmarks in results.pose_world_landmarks:
            landmark_data = []
            for landmark in person_landmarks:
                landmark_data.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "confidence": float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
                })
            all_people_landmarks.append(landmark_data)

        return all_people_landmarks if all_people_landmarks else None

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw pose landmarks on the frame for all detected people

        Args:
            frame: Input image frame
            results: MediaPipe pose detection results

        Returns:
            Frame with drawn landmarks
        """
        if not results or not results.pose_landmarks:
            return frame

        # Draw landmarks for each detected person
        for person_idx, person_landmarks in enumerate(results.pose_landmarks):
            # Use different colors for different people
            if person_idx == 0:
                # Person 1: Green
                landmark_color = (0, 255, 0)
                connection_color = (0, 200, 0)
            elif person_idx == 1:
                # Person 2: Blue
                landmark_color = (255, 0, 0)
                connection_color = (200, 0, 0)
            else:
                # Additional people: Yellow
                landmark_color = (0, 255, 255)
                connection_color = (0, 200, 200)

            # Convert to normalized landmarks format for drawing
            from mediapipe.framework.formats import landmark_pb2
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in person_landmarks
            ])

            self.drawing_utils.draw_landmarks(
                frame,
                pose_landmarks_proto,
                self.pose_connections,
                self.drawing_utils.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                self.drawing_utils.DrawingSpec(color=connection_color, thickness=2)
            )

        return frame

    def get_landmark_by_name(self, results: Any, landmark_name: str, person_idx: int = 0) -> Optional[Dict]:
        """
        Get specific landmark by name for a specific person

        Args:
            results: MediaPipe pose detection results
            landmark_name: Name of the landmark (e.g., 'LEFT_WRIST', 'RIGHT_SHOULDER')
            person_idx: Index of the person (0 = first person, 1 = second person)

        Returns:
            Landmark dictionary or None
        """
        if not results or not results.pose_world_landmarks:
            return None

        if person_idx >= len(results.pose_world_landmarks):
            return None

        try:
            landmark_index = getattr(self.pose_landmarks_class, landmark_name)
            person_landmarks = results.pose_world_landmarks[person_idx]
            landmark = person_landmarks[landmark_index]
            return {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "confidence": float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
            }
        except (AttributeError, IndexError) as e:
            print(f"❌ Error getting landmark {landmark_name} for person {person_idx}: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe Pose (tasks API)",
            "type": "pose",
            "confidence_threshold": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence,
            "num_landmarks": 33,  # MediaPipe pose has 33 landmarks
            "num_poses": self.num_poses,
            "model_path": self.model_path
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.close()
