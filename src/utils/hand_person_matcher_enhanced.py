"""
Enhanced Hand-to-Person Spatial Matching with Temporal Fallback
Implements multi-tier matching strategy to preserve all detection data

Features:
- Returns tuples (matched, unmatched) for consistent API
- Preserves handedness and confidence in unmatched data
- Consistent data structures for all person IDs
- Multi-tier matching: strict current frame, relaxed temporal, unmatched fallback
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from src.utils.hand_person_matcher import (
    calculate_distance,
    get_hand_center,
    get_wrist_positions,
    get_nose_position
)


class HandPersonMatcherEnhanced:
    """
    Enhanced matcher with multi-tier matching strategy:
    - Tier 1: Current frame strict matching
    - Tier 2: Temporal buffer relaxed matching
    - Tier 3: Return unmatched detections separately (for person_id=-1 storage)
    """

    def __init__(self,
                 strict_threshold: float = 0.3,
                 relaxed_threshold: float = 0.5,
                 temporal_lookback: int = 30):
        """
        Args:
            strict_threshold: Threshold for current frame matching
            relaxed_threshold: Threshold for temporal buffer matching
            temporal_lookback: Number of frames to search in history
        """
        self.strict_threshold = strict_threshold
        self.relaxed_threshold = relaxed_threshold
        self.temporal_lookback = temporal_lookback

    def match_hands_with_temporal_fallback(
        self,
        detected_hands: List[Dict],
        current_person_poses: Dict[int, List[Dict]],  # {stable_person_id: pose_landmarks}
        pose_history: List,  # List[TemporalPoseBuffer]
        hand_detector: any = None
    ) -> Tuple[Dict[int, Dict[str, Optional[List[Dict]]]], List[Dict]]:
        """
        Match hands to people with temporal fallback

        Returns:
            Tuple of (matched_hands, unmatched_hands):
            - matched_hands: {
                0: {'left_hand': [...], 'right_hand': [...]},
                1: {'left_hand': [...], 'right_hand': [...]}
              }
            - unmatched_hands: [
                {'landmarks': [...], 'handedness': 'Left', 'confidence': 0.95},
                {'landmarks': [...], 'handedness': 'Right', 'confidence': 0.87}
              ]
        """
        # Initialize result structure for current people
        person_hands = {}
        for person_id in current_person_poses.keys():
            person_hands[person_id] = {
                'left_hand': None,
                'right_hand': None
            }

        if not detected_hands:
            return person_hands, []

        # Extract wrist positions for current people
        current_wrists = self._extract_wrists_from_poses(current_person_poses)

        # Track which hands have been matched
        unmatched_hand_indices = list(range(len(detected_hands)))

        # ============================================================
        # TIER 1: Current Frame Strict Matching (threshold=0.3)
        # ============================================================
        for hand_idx in list(unmatched_hand_indices):
            hand = detected_hands[hand_idx]
            hand_landmarks = hand.get('landmarks', [])

            if not hand_landmarks:
                unmatched_hand_indices.remove(hand_idx)
                continue

            hand_center = get_hand_center(hand_landmarks)
            if not hand_center:
                unmatched_hand_indices.remove(hand_idx)
                continue

            # Find closest wrist
            best_match = self._find_closest_wrist(
                hand_center,
                current_wrists,
                threshold=self.strict_threshold
            )

            if best_match:
                person_id, hand_side = best_match
                # Only assign if this person doesn't already have a hand on this side
                if person_hands[person_id][hand_side] is None:
                    person_hands[person_id][hand_side] = hand_landmarks
                    unmatched_hand_indices.remove(hand_idx)

        # If all hands matched, we're done!
        if not unmatched_hand_indices:
            return person_hands, []

        # ============================================================
        # TIER 2: Temporal Buffer Relaxed Matching (threshold=0.5)
        # ============================================================
        if pose_history:
            # Extract historical wrist positions
            historical_wrists = self._extract_historical_wrists(
                pose_history,
                lookback_frames=self.temporal_lookback
            )

            for hand_idx in list(unmatched_hand_indices):
                hand = detected_hands[hand_idx]
                hand_landmarks = hand.get('landmarks', [])
                hand_center = get_hand_center(hand_landmarks)

                if not hand_center:
                    continue

                # Search through history
                best_match = self._find_closest_historical_wrist(
                    hand_center,
                    historical_wrists,
                    threshold=self.relaxed_threshold
                )

                if best_match:
                    person_id, hand_side = best_match

                    # Initialize this person if not in current frame
                    if person_id not in person_hands:
                        person_hands[person_id] = {
                            'left_hand': None,
                            'right_hand': None
                        }

                    # Assign hand if slot is free
                    if person_hands[person_id][hand_side] is None:
                        person_hands[person_id][hand_side] = hand_landmarks
                        unmatched_hand_indices.remove(hand_idx)

        # ============================================================
        # TIER 3: Return Unmatched Hands with Full Metadata
        # ============================================================
        unmatched_hands = []
        for hand_idx in unmatched_hand_indices:
            hand_data = detected_hands[hand_idx]
            unmatched_hands.append({
                'landmarks': hand_data.get('landmarks'),
                'handedness': hand_data.get('handedness', 'Unknown'),
                'confidence': hand_data.get('confidence', 1.0)
            })

        return person_hands, unmatched_hands

    def match_faces_with_temporal_fallback(
        self,
        face_bboxes: List[Dict],
        facemesh_landmarks_list: List[List[Dict]],
        current_person_poses: Dict[int, List[Dict]],
        pose_history: List
    ) -> Tuple[Dict[int, Optional[Dict]], List[Dict]]:
        """
        Match faces to people with temporal fallback

        Returns:
            Tuple of (matched_faces, unmatched_faces):
            - matched_faces: {
                0: {'bbox': {...}, 'landmarks': [...]},
                1: {'bbox': {...}, 'landmarks': [...]}
              }
            - unmatched_faces: [
                {'bbox': {...}, 'landmarks': [...]},
                {'bbox': {...}, 'landmarks': [...]}
              ]
        """
        person_faces = {}
        for person_id in current_person_poses.keys():
            person_faces[person_id] = None

        if not face_bboxes:
            return person_faces, []

        # Extract nose positions for current people
        current_noses = self._extract_noses_from_poses(current_person_poses)

        unmatched_face_indices = list(range(len(face_bboxes)))

        # ============================================================
        # TIER 1: Current Frame Strict Matching
        # ============================================================
        for face_idx in list(unmatched_face_indices):
            bbox = face_bboxes[face_idx]
            face_center = {
                'x': bbox.get('x', 0) + bbox.get('w', 0) / 2,
                'y': bbox.get('y', 0) + bbox.get('h', 0) / 2
            }

            # Find closest nose
            best_match = self._find_closest_nose(
                face_center,
                current_noses,
                threshold=self.strict_threshold
            )

            if best_match:
                person_id = best_match
                if person_faces[person_id] is None:
                    landmarks = facemesh_landmarks_list[face_idx] if face_idx < len(facemesh_landmarks_list) else None
                    person_faces[person_id] = {
                        'bbox': bbox,
                        'landmarks': landmarks
                    }
                    unmatched_face_indices.remove(face_idx)

        # ============================================================
        # TIER 2: Temporal Buffer Relaxed Matching
        # ============================================================
        if unmatched_face_indices and pose_history:
            historical_noses = self._extract_historical_noses(
                pose_history,
                lookback_frames=self.temporal_lookback
            )

            for face_idx in list(unmatched_face_indices):
                bbox = face_bboxes[face_idx]
                face_center = {
                    'x': bbox.get('x', 0) + bbox.get('w', 0) / 2,
                    'y': bbox.get('y', 0) + bbox.get('h', 0) / 2
                }

                best_match = self._find_closest_historical_nose(
                    face_center,
                    historical_noses,
                    threshold=self.relaxed_threshold
                )

                if best_match:
                    person_id = best_match
                    if person_id not in person_faces:
                        person_faces[person_id] = None

                    if person_faces[person_id] is None:
                        landmarks = facemesh_landmarks_list[face_idx] if face_idx < len(facemesh_landmarks_list) else None
                        person_faces[person_id] = {
                            'bbox': bbox,
                            'landmarks': landmarks
                        }
                        unmatched_face_indices.remove(face_idx)

        # ============================================================
        # TIER 3: Return Unmatched Faces with Full Data
        # ============================================================
        unmatched_faces = []
        for face_idx in unmatched_face_indices:
            bbox = face_bboxes[face_idx]
            landmarks = facemesh_landmarks_list[face_idx] if face_idx < len(facemesh_landmarks_list) else None
            unmatched_faces.append({
                'bbox': bbox,
                'landmarks': landmarks
            })

        return person_faces, unmatched_faces

    def _extract_wrists_from_poses(
        self,
        person_poses: Dict[int, List[Dict]]
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Extract wrist positions from current poses

        Returns:
            {person_id: {'left_wrist': {...}, 'right_wrist': {...}}}
        """
        wrists = {}
        for person_id, pose_landmarks in person_poses.items():
            left_wrist, right_wrist = get_wrist_positions(pose_landmarks)
            wrists[person_id] = {
                'left_wrist': left_wrist,
                'right_wrist': right_wrist
            }
        return wrists

    def _extract_noses_from_poses(
        self,
        person_poses: Dict[int, List[Dict]]
    ) -> Dict[int, Dict]:
        """Extract nose positions from current poses"""
        noses = {}
        for person_id, pose_landmarks in person_poses.items():
            nose = get_nose_position(pose_landmarks)
            if nose:
                noses[person_id] = nose
        return noses

    def _extract_historical_wrists(
        self,
        pose_history: List,
        lookback_frames: int
    ) -> List[Dict]:
        """
        Extract wrist positions from historical poses

        Returns:
            List of {person_id, frame_number, left_wrist, right_wrist}
        """
        historical_wrists = []

        for buffer in pose_history[:lookback_frames]:
            for track in buffer.person_tracks:
                left_wrist, right_wrist = get_wrist_positions(track.pose_landmarks)
                historical_wrists.append({
                    'person_id': track.person_id,
                    'frame_number': buffer.frame_number,
                    'left_wrist': left_wrist,
                    'right_wrist': right_wrist
                })

        return historical_wrists

    def _extract_historical_noses(
        self,
        pose_history: List,
        lookback_frames: int
    ) -> List[Dict]:
        """Extract nose positions from historical poses"""
        historical_noses = []

        for buffer in pose_history[:lookback_frames]:
            for track in buffer.person_tracks:
                nose = get_nose_position(track.pose_landmarks)
                if nose:
                    historical_noses.append({
                        'person_id': track.person_id,
                        'frame_number': buffer.frame_number,
                        'nose': nose
                    })

        return historical_noses

    def _find_closest_wrist(
        self,
        hand_center: Dict,
        wrists: Dict[int, Dict[str, Dict]],
        threshold: float
    ) -> Optional[Tuple[int, str]]:
        """Find closest wrist to hand center"""
        min_distance = float('inf')
        best_match = None

        for person_id, wrist_data in wrists.items():
            # Check left wrist
            if wrist_data['left_wrist']:
                dist = calculate_distance(hand_center, wrist_data['left_wrist'])
                if dist < min_distance and dist < threshold:
                    min_distance = dist
                    best_match = (person_id, 'left_hand')

            # Check right wrist
            if wrist_data['right_wrist']:
                dist = calculate_distance(hand_center, wrist_data['right_wrist'])
                if dist < min_distance and dist < threshold:
                    min_distance = dist
                    best_match = (person_id, 'right_hand')

        return best_match

    def _find_closest_historical_wrist(
        self,
        hand_center: Dict,
        historical_wrists: List[Dict],
        threshold: float
    ) -> Optional[Tuple[int, str]]:
        """Find closest historical wrist"""
        min_distance = float('inf')
        best_match = None

        for wrist_data in historical_wrists:
            # Check left wrist
            if wrist_data['left_wrist']:
                dist = calculate_distance(hand_center, wrist_data['left_wrist'])
                if dist < min_distance and dist < threshold:
                    min_distance = dist
                    best_match = (wrist_data['person_id'], 'left_hand')

            # Check right wrist
            if wrist_data['right_wrist']:
                dist = calculate_distance(hand_center, wrist_data['right_wrist'])
                if dist < min_distance and dist < threshold:
                    min_distance = dist
                    best_match = (wrist_data['person_id'], 'right_hand')

        return best_match

    def _find_closest_nose(
        self,
        face_center: Dict,
        noses: Dict[int, Dict],
        threshold: float
    ) -> Optional[int]:
        """Find closest nose to face center"""
        min_distance = float('inf')
        best_person_id = None

        for person_id, nose in noses.items():
            dist = calculate_distance(face_center, nose)
            if dist < min_distance and dist < threshold:
                min_distance = dist
                best_person_id = person_id

        return best_person_id

    def _find_closest_historical_nose(
        self,
        face_center: Dict,
        historical_noses: List[Dict],
        threshold: float
    ) -> Optional[int]:
        """Find closest historical nose"""
        min_distance = float('inf')
        best_person_id = None

        for nose_data in historical_noses:
            dist = calculate_distance(face_center, nose_data['nose'])
            if dist < min_distance and dist < threshold:
                min_distance = dist
                best_person_id = nose_data['person_id']

        return best_person_id


def convert_hand_results_to_detected_hands(
    hand_results: any,
    hand_detector: any
) -> List[Dict]:
    """
    Convert MediaPipe hand results to detected hands list

    Returns:
        List of {'landmarks': [...], 'handedness': 'Left'/'Right', 'confidence': float}
    """
    detected_hands = []

    if hand_results and hasattr(hand_results, 'hand_landmarks'):
        for hand_idx, landmarks in enumerate(hand_results.hand_landmarks):
            if hand_detector and hasattr(hand_detector, 'convert_to_dict'):
                # Create mock result object
                class SingleHandResult:
                    def __init__(self, landmarks, world_landmarks, handedness):
                        self.hand_landmarks = [landmarks]
                        self.hand_world_landmarks = [world_landmarks] if world_landmarks else None
                        self.handedness = [handedness] if handedness else None

                world_landmarks = hand_results.hand_world_landmarks[hand_idx] if hand_results.hand_world_landmarks else None
                handedness = hand_results.handedness[hand_idx] if hand_results.handedness else None

                single_result = SingleHandResult(landmarks, world_landmarks, handedness)
                landmarks_dict = hand_detector.convert_to_dict(single_result)

                # MediaPipe Tasks API: handedness is already a list of Category objects
                handedness_label = 'Unknown'
                confidence = 1.0
                if handedness and len(handedness) > 0:
                    # handedness is already a list: [Category]
                    handedness_label = handedness[0].category_name
                    confidence = handedness[0].score

                detected_hands.append({
                    'landmarks': landmarks_dict,
                    'handedness': handedness_label,
                    'confidence': confidence
                })

    return detected_hands
