import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Dict, Any
from .base_hand_detector import BaseHandDetector


class MediaPipeHandDetector(BaseHandDetector):
    """MediaPipe hand detection model using new tasks API with multi-hand support"""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 num_hands: int = 4, model_path: str = None):
        """
        Initialize MediaPipe Hand detector with new tasks API

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            num_hands: Maximum number of hands to detect (default: 4, supports 2 people)
            model_path: Path to .task model file (default: hand_landmarker.task)
        """
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands

        # Default model path
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            model_path = os.path.join(project_root, "src", "checkpoints", "hand_landmarker.task")

        self.model_path = model_path

        # Initialize model with new tasks API
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.model = vision.HandLandmarker.create_from_options(options)
            print(f"✅ MediaPipe Hands (tasks API) loaded: {num_hands} hands, model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Failed to load MediaPipe Hand model: {e}")
            print(f"   Download model from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models")
            self.model = None

        # Drawing utilities
        self.drawing_utils = mp.solutions.drawing_utils
        self.hand_landmarks_class = mp.solutions.hands.HandLandmark
        # Use hand connections from the legacy API (compatible with drawing_utils)
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

        # Frame counter for video processing
        self.frame_timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect hands in the frame (supports multiple hands)

        Args:
            frame: Input image frame

        Returns:
            MediaPipe hand detection results or None
        """
        if self.model is None:
            print("❌ Hand model is None - model did not load properly!")
            return None

        try:
            # Convert to RGB and create MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands in the image
            results = self.model.detect(mp_image)

            return results
        except Exception as e:
            print(f"❌ Hand detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_to_dict(self, results: Any) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Convert MediaPipe hand landmarks to dictionary format
        Returns (left_hand, right_hand) for first person for backward compatibility

        Args:
            results: MediaPipe hand detection results

        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        left_hand = None
        right_hand = None

        if results and results.hand_world_landmarks and results.handedness:
            # Process hands and separate by handedness
            for i, (hand_landmarks, handedness) in enumerate(zip(results.hand_world_landmarks, results.handedness)):
                hand_data = []
                for landmark in hand_landmarks:
                    hand_data.append({
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "confidence": 1.0
                    })

                # Determine if left or right hand
                hand_label = handedness[0].category_name  # "Left" or "Right"

                # Assign to first left/right hand found (for backward compatibility)
                if hand_label == "Left" and left_hand is None:
                    left_hand = hand_data
                elif hand_label == "Right" and right_hand is None:
                    right_hand = hand_data

        return left_hand, right_hand

    def convert_to_dict_multi_hand(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert MediaPipe hand landmarks to dictionary format for all detected hands

        Args:
            results: MediaPipe hand detection results

        Returns:
            List of hand dictionaries with handedness info
            Format: [{"handedness": "Left", "landmarks": [...]}, ...]
        """
        if not results or not results.hand_world_landmarks or not results.handedness:
            return None

        all_hands = []

        for hand_landmarks, handedness in zip(results.hand_world_landmarks, results.handedness):
            landmark_data = []
            for landmark in hand_landmarks:
                landmark_data.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "confidence": 1.0
                })

            all_hands.append({
                "handedness": handedness[0].category_name,  # "Left" or "Right"
                "confidence": handedness[0].score,
                "landmarks": landmark_data
            })

        return all_hands if all_hands else None

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand landmarks on the frame for all detected hands

        Args:
            frame: Input image frame
            results: MediaPipe hand detection results

        Returns:
            Frame with drawn landmarks
        """
        if not results or not results.hand_landmarks:
            return frame

        # Draw landmarks for each detected hand
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Use different colors based on handedness if available
            if results.handedness and hand_idx < len(results.handedness):
                hand_label = results.handedness[hand_idx][0].category_name
                if hand_label == "Left":
                    landmark_color = (0, 255, 0)  # Green for left hand
                    connection_color = (0, 200, 0)
                else:
                    landmark_color = (255, 0, 0)  # Blue for right hand
                    connection_color = (200, 0, 0)
            else:
                landmark_color = (255, 255, 0)  # Yellow if unknown
                connection_color = (200, 200, 0)

            # Convert to normalized landmarks format for drawing
            from mediapipe.framework.formats import landmark_pb2
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
            ])

            self.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks_proto,
                self.hand_connections,
                self.drawing_utils.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                self.drawing_utils.DrawingSpec(color=connection_color, thickness=2)
            )

        return frame

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe Hands (tasks API)",
            "type": "hand",
            "confidence_threshold": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence,
            "num_hands": self.num_hands,
            "model_path": self.model_path
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.close()
