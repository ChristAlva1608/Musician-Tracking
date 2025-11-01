"""
Enhanced Person Tracker with Temporal History
Maintains consistent person IDs across frames and stores pose history for spatial matching
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PersonTrack:
    """Represents a tracked person across frames"""
    person_id: int
    last_seen_frame: int
    pose_landmarks: List[Dict]
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    age: int = 0  # How many frames this track has existed


@dataclass
class TemporalPoseBuffer:
    """Stores pose detection history for temporal matching"""
    frame_number: int
    person_tracks: List[PersonTrack]


class PersonTrackerWithHistory:
    """
    Person tracker that maintains:
    1. Consistent person IDs across frames using centroid tracking
    2. Temporal buffer of pose detections for historical matching
    """

    def __init__(self,
                 max_history_frames: int = 30,
                 max_disappeared_frames: int = 15,
                 distance_threshold: float = 0.15):
        """
        Args:
            max_history_frames: Number of frames to keep in history buffer
            max_disappeared_frames: Max frames a person can disappear before ID is recycled
            distance_threshold: Max distance for person matching across frames
        """
        self.max_history_frames = max_history_frames
        self.max_disappeared_frames = max_disappeared_frames
        self.distance_threshold = distance_threshold

        # Active tracks
        self.active_tracks: Dict[int, PersonTrack] = {}
        self.next_person_id = 0

        # Temporal buffer (FIFO queue)
        self.pose_history = deque(maxlen=max_history_frames)

        # Track disappeared persons
        self.disappeared = {}  # person_id -> frames_disappeared

    def update(self,
               frame_number: int,
               pose_landmarks_multi: List[List[Dict]]) -> Dict[int, List[Dict]]:
        """
        Update tracker with current frame detections

        Args:
            frame_number: Current frame number
            pose_landmarks_multi: List of detected pose landmarks

        Returns:
            Dictionary mapping consistent person_id to pose landmarks
            {stable_person_id: pose_landmarks}
        """
        if not pose_landmarks_multi:
            # Increment disappeared counter for all active tracks
            for person_id in list(self.active_tracks.keys()):
                if person_id not in self.disappeared:
                    self.disappeared[person_id] = 0
                self.disappeared[person_id] += 1

                # Remove tracks that have been gone too long
                if self.disappeared[person_id] > self.max_disappeared_frames:
                    del self.active_tracks[person_id]
                    del self.disappeared[person_id]

            # ✅ FIX: Add empty frame to history for temporal continuity
            self.pose_history.append(TemporalPoseBuffer(
                frame_number=frame_number,
                person_tracks=[]  # Empty list when no poses detected
            ))

            return {}

        # Calculate centroids and bboxes for current detections
        current_centroids = []
        current_bboxes = []
        for pose_landmarks in pose_landmarks_multi:
            centroid = self._calculate_centroid(pose_landmarks)
            bbox = self._calculate_bbox(pose_landmarks)
            current_centroids.append(centroid)
            current_bboxes.append(bbox)

        # If no existing tracks, register all as new
        if len(self.active_tracks) == 0:
            person_id_mapping = {}
            new_tracks = []

            for idx, (pose_landmarks, centroid, bbox) in enumerate(
                zip(pose_landmarks_multi, current_centroids, current_bboxes)):
                person_id = self.next_person_id
                self.next_person_id += 1

                track = PersonTrack(
                    person_id=person_id,
                    last_seen_frame=frame_number,
                    pose_landmarks=pose_landmarks,
                    centroid=centroid,
                    bbox=bbox,
                    age=1
                )
                self.active_tracks[person_id] = track
                new_tracks.append(track)
                person_id_mapping[person_id] = pose_landmarks

            # Add to history
            self.pose_history.append(TemporalPoseBuffer(
                frame_number=frame_number,
                person_tracks=new_tracks
            ))

            return person_id_mapping

        # Match current detections to existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            current_centroids, pose_landmarks_multi
        )

        person_id_mapping = {}
        updated_tracks = []

        # Update matched tracks
        for detection_idx, track_id in matched_pairs:
            track = self.active_tracks[track_id]
            track.last_seen_frame = frame_number
            track.pose_landmarks = pose_landmarks_multi[detection_idx]
            track.centroid = current_centroids[detection_idx]
            track.bbox = current_bboxes[detection_idx]
            track.age += 1

            person_id_mapping[track_id] = pose_landmarks_multi[detection_idx]
            updated_tracks.append(track)

            # Remove from disappeared if it was there
            if track_id in self.disappeared:
                del self.disappeared[track_id]

        # Register new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            person_id = self.next_person_id
            self.next_person_id += 1

            track = PersonTrack(
                person_id=person_id,
                last_seen_frame=frame_number,
                pose_landmarks=pose_landmarks_multi[detection_idx],
                centroid=current_centroids[detection_idx],
                bbox=current_bboxes[detection_idx],
                age=1
            )
            self.active_tracks[person_id] = track
            updated_tracks.append(track)
            person_id_mapping[person_id] = pose_landmarks_multi[detection_idx]

        # Mark unmatched tracks as disappeared
        for track_id in unmatched_tracks:
            if track_id not in self.disappeared:
                self.disappeared[track_id] = 0
            self.disappeared[track_id] += 1

            # Remove if disappeared too long
            if self.disappeared[track_id] > self.max_disappeared_frames:
                del self.active_tracks[track_id]
                del self.disappeared[track_id]

        # Add to history
        self.pose_history.append(TemporalPoseBuffer(
            frame_number=frame_number,
            person_tracks=updated_tracks.copy()
        ))

        return person_id_mapping

    def get_historical_poses(self,
                            lookback_frames: Optional[int] = None) -> List[TemporalPoseBuffer]:
        """
        Get pose history for temporal matching

        Args:
            lookback_frames: Number of frames to look back (None = all available)

        Returns:
            List of TemporalPoseBuffer from most recent to oldest
        """
        if lookback_frames is None:
            return list(reversed(self.pose_history))
        else:
            return list(reversed(list(self.pose_history)[-lookback_frames:]))

    def _calculate_centroid(self, pose_landmarks: List[Dict]) -> Tuple[float, float]:
        """Calculate centroid of pose using torso keypoints"""
        if not pose_landmarks or len(pose_landmarks) < 12:
            return (0.0, 0.0)

        # Use torso keypoints (shoulders and hips) for stable centroid
        # MediaPipe indices: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        torso_indices = [11, 12, 23, 24]

        x_coords = []
        y_coords = []
        for idx in torso_indices:
            if idx < len(pose_landmarks):
                x_coords.append(pose_landmarks[idx].get('x', 0))
                y_coords.append(pose_landmarks[idx].get('y', 0))

        if not x_coords:
            return (0.0, 0.0)

        return (np.mean(x_coords), np.mean(y_coords))

    def _calculate_bbox(self, pose_landmarks: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate bounding box of pose"""
        if not pose_landmarks:
            return (0.0, 0.0, 0.0, 0.0)

        x_coords = [lm.get('x', 0) for lm in pose_landmarks]
        y_coords = [lm.get('y', 0) for lm in pose_landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _match_detections_to_tracks(self,
                                    current_centroids: List[Tuple[float, float]],
                                    pose_landmarks_multi: List[List[Dict]]) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match current detections to existing tracks using Hungarian algorithm

        Returns:
            matched_pairs: List of (detection_idx, track_id)
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track IDs
        """
        if len(current_centroids) == 0:
            return [], [], list(self.active_tracks.keys())

        # Compute distance matrix
        track_ids = list(self.active_tracks.keys())
        track_centroids = [self.active_tracks[tid].centroid for tid in track_ids]

        distance_matrix = np.zeros((len(current_centroids), len(track_centroids)))

        for i, current_centroid in enumerate(current_centroids):
            for j, track_centroid in enumerate(track_centroids):
                dx = current_centroid[0] - track_centroid[0]
                dy = current_centroid[1] - track_centroid[1]
                distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

        # Simple greedy matching (could use scipy.optimize.linear_sum_assignment for optimal)
        matched_pairs = []
        unmatched_detections = set(range(len(current_centroids)))
        unmatched_tracks = set(range(len(track_ids)))

        # Greedy: match smallest distances first
        while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            min_dist = np.inf
            best_det_idx = None
            best_track_idx = None

            for det_idx in unmatched_detections:
                for track_idx in unmatched_tracks:
                    if distance_matrix[det_idx, track_idx] < min_dist:
                        min_dist = distance_matrix[det_idx, track_idx]
                        best_det_idx = det_idx
                        best_track_idx = track_idx

            if min_dist < self.distance_threshold:
                matched_pairs.append((best_det_idx, track_ids[best_track_idx]))
                unmatched_detections.remove(best_det_idx)
                unmatched_tracks.remove(best_track_idx)
            else:
                break

        unmatched_track_ids = [track_ids[idx] for idx in unmatched_tracks]

        return matched_pairs, list(unmatched_detections), unmatched_track_ids
