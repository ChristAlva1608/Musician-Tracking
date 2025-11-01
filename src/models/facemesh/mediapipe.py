import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from typing import Optional, List, Dict, Any
from .base_facemesh_detector import BaseFaceMeshDetector


class MediaPipeFaceMeshDetector(BaseFaceMeshDetector):
    """MediaPipe face detection model using new tasks API with multi-face support"""

    def __init__(self, model_path: str = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 num_faces: int = 2):
        """
        Initialize MediaPipe FaceMesh detector with new tasks API

        Args:
            model_path: Path to .task model file (default: face_landmarker.task)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            num_faces: Maximum number of faces to detect (default: 2)
        """
        super().__init__(confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_faces = num_faces

        # Use absolute path to checkpoints
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            model_path = os.path.join(project_root, "src", "checkpoints", "face_landmarker.task")

        self.model_path = model_path
        self.model = None

        try:
            # Use new FaceLandmarker API with IMAGE mode
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.model = vision.FaceLandmarker.create_from_options(options)
            print(f"✅ MediaPipe FaceMesh (tasks API) loaded: {num_faces} faces, model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Failed to load MediaPipe FaceMesh model: {e}")
            print(f"   Download model from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models")
            self.model = None

        # Drawing utilities
        self.drawing_utils = mp.solutions.drawing_utils
        # Use FaceLandmarksConnections from MediaPipe Tasks API
        from mediapipe.tasks.python.vision import FaceLandmarksConnections
        self.face_mesh_connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

        # Frame counter for video processing
        self.frame_timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect face landmarks in the frame (supports multiple faces)

        Args:
            frame: Input image frame

        Returns:
            MediaPipe face detection results or None
        """
        if self.model is None:
            return None

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Detect face landmarks in the image
            results = self.model.detect(mp_image)
            return results
        except Exception as e:
            print(f"❌ MediaPipe face detection failed: {e}")
            return None

    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """
        Convert MediaPipe face landmarks to dictionary format
        Returns first face's landmarks for backward compatibility

        Args:
            results: MediaPipe face detection results

        Returns:
            List of face landmark dictionaries for first face, or None
        """
        if not results or not results.face_landmarks:
            return None

        try:
            # Get landmarks from the first detected face
            if len(results.face_landmarks) > 0:
                face_landmarks = results.face_landmarks[0]
                landmark_data = []

                for landmark in face_landmarks:
                    landmark_data.append({
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "confidence": 1.0  # MediaPipe FaceLandmarker doesn't provide per-landmark confidence
                    })

                return landmark_data

        except Exception as e:
            print(f"❌ Error converting face landmarks: {e}")

        return None

    def convert_to_dict_multi_face(self, results: Any) -> Optional[List[List[Dict]]]:
        """
        Convert MediaPipe face landmarks to dictionary format for all detected faces

        Args:
            results: MediaPipe face detection results

        Returns:
            List of faces, each containing list of landmark dictionaries
            Format: [[face1_landmarks], [face2_landmarks], ...]
        """
        if not results or not results.face_landmarks:
            return None

        all_faces_landmarks = []

        for face_landmarks in results.face_landmarks:
            landmark_data = []
            for landmark in face_landmarks:
                landmark_data.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "confidence": 1.0
                })
            all_faces_landmarks.append(landmark_data)

        return all_faces_landmarks if all_faces_landmarks else None

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw face landmarks on the frame for all detected faces

        Args:
            frame: Input image frame
            results: MediaPipe face detection results

        Returns:
            Frame with drawn landmarks
        """
        if not results or not results.face_landmarks:
            return frame

        # Draw landmarks for each detected face
        for face_idx, face_landmarks in enumerate(results.face_landmarks):
            # Use different colors for different faces
            if face_idx == 0:
                # Face 1: Green
                landmark_color = (0, 255, 0)
                connection_color = (0, 200, 0)
            elif face_idx == 1:
                # Face 2: Blue
                landmark_color = (255, 0, 0)
                connection_color = (200, 0, 0)
            else:
                # Additional faces: Yellow
                landmark_color = (0, 255, 255)
                connection_color = (0, 200, 200)

            # Convert to normalized landmarks format for drawing
            from mediapipe.framework.formats import landmark_pb2
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks
            ])

            self.drawing_utils.draw_landmarks(
                frame,
                face_landmarks_proto,
                self.face_mesh_connections,
                self.drawing_utils.DrawingSpec(color=landmark_color, thickness=1, circle_radius=1),
                self.drawing_utils.DrawingSpec(color=connection_color, thickness=1)
            )

        return frame

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "MediaPipe FaceMesh (tasks API)",
            "type": "facemesh",
            "confidence_threshold": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence,
            "num_faces": self.num_faces,
            "model_path": self.model_path
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.close()
