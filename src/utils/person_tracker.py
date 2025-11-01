#!/usr/bin/env python3
"""
Person Tracking Utilities
Provides functions to maintain consistent person ordering across frames
"""

import numpy as np
from typing import Any, Optional, List, Tuple


def sort_people_by_horizontal_position(pose_results: Any) -> Any:
    """
    Sort detected people by horizontal position (left to right) for consistent indexing.

    This ensures that:
    - Person on the left is always index 0 (typically drawn in green)
    - Person on the right is always index 1 (typically drawn in blue)

    Args:
        pose_results: MediaPipe pose detection results with pose_landmarks

    Returns:
        Sorted pose_results with consistent left-to-right ordering

    Example:
        >>> pose_results = detector.detect(frame)
        >>> pose_results = sort_people_by_horizontal_position(pose_results)
        >>> # Now pose_results.pose_landmarks[0] is always the leftmost person
    """
    if not pose_results:
        return pose_results

    # Check if pose_landmarks exists and has multiple people
    if not hasattr(pose_results, 'pose_landmarks') or not pose_results.pose_landmarks:
        return pose_results

    if len(pose_results.pose_landmarks) <= 1:
        return pose_results  # Only one person, no need to sort

    # Calculate horizontal center position for each person
    people_with_positions = []

    for person_idx, person_landmarks in enumerate(pose_results.pose_landmarks):
        if not person_landmarks or len(person_landmarks) == 0:
            # Handle empty landmarks
            people_with_positions.append((0.5, person_idx))
            continue

        # Use nose landmark (index 0) as primary reference point
        # Fallback to shoulder center if nose not available
        try:
            nose_x = person_landmarks[0].x
        except (IndexError, AttributeError):
            # Fallback: use shoulder center (landmarks 11 and 12)
            try:
                left_shoulder_x = person_landmarks[11].x
                right_shoulder_x = person_landmarks[12].x
                nose_x = (left_shoulder_x + right_shoulder_x) / 2
            except (IndexError, AttributeError):
                # Final fallback: use 0.5 (center)
                nose_x = 0.5

        people_with_positions.append((nose_x, person_idx))

    # Sort by x-position (left to right)
    people_with_positions.sort(key=lambda x: x[0])

    # Extract sorted indices
    sorted_indices = [idx for _, idx in people_with_positions]

    # Reorder pose_landmarks
    sorted_pose_landmarks = [pose_results.pose_landmarks[i] for i in sorted_indices]
    pose_results.pose_landmarks = sorted_pose_landmarks

    # Reorder pose_world_landmarks if available (for 3D coordinates)
    if hasattr(pose_results, 'pose_world_landmarks') and pose_results.pose_world_landmarks:
        sorted_world_landmarks = [pose_results.pose_world_landmarks[i] for i in sorted_indices]
        pose_results.pose_world_landmarks = sorted_world_landmarks

    return pose_results


def calculate_person_center(person_landmarks) -> Tuple[float, float]:
    """
    Calculate the center position of a person based on their landmarks.

    Args:
        person_landmarks: List of pose landmarks for one person

    Returns:
        Tuple of (center_x, center_y) in normalized coordinates [0, 1]
    """
    if not person_landmarks or len(person_landmarks) == 0:
        return (0.5, 0.5)

    try:
        # Use torso landmarks for center calculation (more stable than extremities)
        # MediaPipe Pose landmark indices:
        # 11: left_shoulder, 12: right_shoulder
        # 23: left_hip, 24: right_hip

        left_shoulder = person_landmarks[11]
        right_shoulder = person_landmarks[12]
        left_hip = person_landmarks[23]
        right_hip = person_landmarks[24]

        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

        return (center_x, center_y)

    except (IndexError, AttributeError):
        # Fallback: use first landmark (nose)
        try:
            return (person_landmarks[0].x, person_landmarks[0].y)
        except (IndexError, AttributeError):
            return (0.5, 0.5)


class PersonTracker:
    """
    Advanced person tracker that maintains identity across frames using position tracking.

    This class uses frame-to-frame position matching to ensure consistent person IDs
    even when people cross paths or swap positions.

    Usage:
        tracker = PersonTracker()
        for frame in video:
            pose_results = detector.detect(frame)
            pose_results = tracker.track_and_sort(pose_results)
    """

    def __init__(self, max_distance_threshold: float = 0.3):
        """
        Initialize person tracker.

        Args:
            max_distance_threshold: Maximum normalized distance to consider a match
                                   (default: 0.3 = 30% of frame width/height)
        """
        self.prev_positions: List[Tuple[float, float]] = []
        self.max_distance_threshold = max_distance_threshold
        self.frame_count = 0

    def track_and_sort(self, pose_results: Any) -> Any:
        """
        Track people across frames and maintain consistent ordering.

        Args:
            pose_results: MediaPipe pose detection results

        Returns:
            Sorted pose_results with consistent person IDs
        """
        if not pose_results or not hasattr(pose_results, 'pose_landmarks'):
            return pose_results

        if not pose_results.pose_landmarks:
            # Reset tracking if no detections
            self.prev_positions = []
            return pose_results

        current_positions = [
            calculate_person_center(person_landmarks)
            for person_landmarks in pose_results.pose_landmarks
        ]

        if not self.prev_positions or len(self.prev_positions) != len(current_positions):
            # First frame or number of people changed - use spatial sorting
            self.prev_positions = current_positions
            self.frame_count = 0
            return sort_people_by_horizontal_position(pose_results)

        # Match current detections to previous using distance
        matched_indices = self._match_by_distance(self.prev_positions, current_positions)

        # Reorder results based on matches
        sorted_pose_landmarks = [pose_results.pose_landmarks[i] for i in matched_indices]
        pose_results.pose_landmarks = sorted_pose_landmarks

        if hasattr(pose_results, 'pose_world_landmarks') and pose_results.pose_world_landmarks:
            sorted_world_landmarks = [pose_results.pose_world_landmarks[i] for i in matched_indices]
            pose_results.pose_world_landmarks = sorted_world_landmarks

        # Update previous positions
        self.prev_positions = [current_positions[i] for i in matched_indices]
        self.frame_count += 1

        return pose_results

    def _match_by_distance(self, prev_positions: List[Tuple[float, float]],
                          curr_positions: List[Tuple[float, float]]) -> List[int]:
        """
        Match current detections to previous using minimum distance (greedy for 2 people).

        Args:
            prev_positions: List of (x, y) positions from previous frame
            curr_positions: List of (x, y) positions from current frame

        Returns:
            List of indices mapping curr_positions to prev_positions order
        """
        n_people = len(curr_positions)

        if n_people == 1:
            return [0]

        if n_people == 2:
            # For 2 people, simple greedy matching
            dist_00 = self._euclidean_distance(prev_positions[0], curr_positions[0])
            dist_01 = self._euclidean_distance(prev_positions[0], curr_positions[1])
            dist_10 = self._euclidean_distance(prev_positions[1], curr_positions[0])
            dist_11 = self._euclidean_distance(prev_positions[1], curr_positions[1])

            # Cost of maintaining order vs swapping
            cost_maintain = dist_00 + dist_11
            cost_swap = dist_01 + dist_10

            if cost_maintain < cost_swap:
                return [0, 1]  # Keep order
            else:
                return [1, 0]  # Swap

        # For more than 2 people, use simple greedy assignment
        # (Can be upgraded to Hungarian algorithm if needed)
        used_indices = set()
        matched_indices = []

        for prev_idx in range(n_people):
            best_curr_idx = None
            best_distance = float('inf')

            for curr_idx in range(n_people):
                if curr_idx in used_indices:
                    continue

                distance = self._euclidean_distance(prev_positions[prev_idx], curr_positions[curr_idx])
                if distance < best_distance:
                    best_distance = distance
                    best_curr_idx = curr_idx

            if best_curr_idx is not None:
                matched_indices.append(best_curr_idx)
                used_indices.add(best_curr_idx)
            else:
                # Fallback: maintain index
                matched_indices.append(prev_idx if prev_idx < n_people else 0)

        return matched_indices

    def _euclidean_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two 2D positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reset(self):
        """Reset tracker state (call when starting a new video or scene)."""
        self.prev_positions = []
        self.frame_count = 0
