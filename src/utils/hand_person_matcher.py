"""
Hand-to-Person Spatial Matching Utility

This module provides functions to match detected hands to specific people
based on spatial proximity to their wrist positions from pose landmarks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def calculate_distance(point1: Dict, point2: Dict) -> float:
    """
    Calculate Euclidean distance between two points

    Args:
        point1: Dictionary with 'x', 'y' keys
        point2: Dictionary with 'x', 'y' keys

    Returns:
        Euclidean distance between the points
    """
    # Extract x, y values, handling nested structures
    x1 = point1.get('x', 0)
    y1 = point1.get('y', 0)
    x2 = point2.get('x', 0)
    y2 = point2.get('y', 0)

    # If x or y is a dict (nested structure), extract the value
    if isinstance(x1, dict):
        x1 = x1.get('x', 0) if 'x' in x1 else 0
    if isinstance(y1, dict):
        y1 = y1.get('y', 0) if 'y' in y1 else 0
    if isinstance(x2, dict):
        x2 = x2.get('x', 0) if 'x' in x2 else 0
    if isinstance(y2, dict):
        y2 = y2.get('y', 0) if 'y' in y2 else 0

    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)
    return np.sqrt(dx**2 + dy**2)


def get_hand_center(hand_landmarks: List[Dict]) -> Optional[Dict]:
    """
    Calculate the center point of a hand based on its landmarks

    Args:
        hand_landmarks: List of hand landmark dictionaries with 'x', 'y' keys

    Returns:
        Dictionary with 'x', 'y' coordinates of hand center, or None if invalid
    """
    if not hand_landmarks or len(hand_landmarks) == 0:
        return None

    # Use wrist (landmark 0) as the primary reference point
    # This is more reliable than averaging all landmarks
    if len(hand_landmarks) > 0:
        wrist = hand_landmarks[0]
        # Ensure wrist is a dictionary before calling .get()
        if isinstance(wrist, dict):
            return {'x': wrist.get('x', 0), 'y': wrist.get('y', 0)}
        elif hasattr(wrist, 'x') and hasattr(wrist, 'y'):
            # Handle case where it's an object with x, y attributes
            return {'x': wrist.x, 'y': wrist.y}
        elif isinstance(wrist, (list, tuple)) and len(wrist) >= 2:
            # Handle case where it's a list/tuple of coordinates [x, y]
            return {'x': wrist[0], 'y': wrist[1]}

    return None


def get_wrist_positions(pose_landmarks: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract left and right wrist positions from pose landmarks

    MediaPipe Pose landmark indices:
    - 15: Left wrist
    - 16: Right wrist

    Args:
        pose_landmarks: List of pose landmark dictionaries

    Returns:
        Tuple of (left_wrist, right_wrist) dictionaries with 'x', 'y' keys
    """
    left_wrist = None
    right_wrist = None

    if pose_landmarks and len(pose_landmarks) >= 17:
        # MediaPipe pose landmarks: index 15 = left wrist, 16 = right wrist
        if len(pose_landmarks) > 15:
            lw = pose_landmarks[15]
            if isinstance(lw, dict):
                left_wrist = {'x': lw.get('x', 0), 'y': lw.get('y', 0)}
            elif hasattr(lw, 'x') and hasattr(lw, 'y'):
                left_wrist = {'x': lw.x, 'y': lw.y}
            elif isinstance(lw, (list, tuple)) and len(lw) >= 2:
                left_wrist = {'x': lw[0], 'y': lw[1]}

        if len(pose_landmarks) > 16:
            rw = pose_landmarks[16]
            if isinstance(rw, dict):
                right_wrist = {'x': rw.get('x', 0), 'y': rw.get('y', 0)}
            elif hasattr(rw, 'x') and hasattr(rw, 'y'):
                right_wrist = {'x': rw.x, 'y': rw.y}
            elif isinstance(rw, (list, tuple)) and len(rw) >= 2:
                right_wrist = {'x': rw[0], 'y': rw[1]}

    return left_wrist, right_wrist


def assign_hands_to_people(
    detected_hands: List[Dict],
    pose_landmarks_multi: List[List[Dict]],
    max_distance_threshold: float = 0.3
) -> Dict[int, Dict[str, Optional[List[Dict]]]]:
    """
    Assign detected hands to specific people based on spatial proximity

    Strategy:
    1. For each person, get their left and right wrist positions from pose
    2. For each detected hand, calculate distance to all wrists
    3. Assign hand to the person with the closest wrist
    4. Ensure each person gets maximum 2 hands (left and right)

    Args:
        detected_hands: List of hand detection dictionaries, each containing:
            - 'landmarks': List of hand landmarks
            - 'handedness': 'Left' or 'Right' (from detector's perspective, mirrored)
            - 'confidence': Detection confidence score
        pose_landmarks_multi: List of pose landmarks for each person
            - Each element is a list of pose landmarks for one person
        max_distance_threshold: Maximum distance to consider a hand match (normalized 0-1)

    Returns:
        Dictionary mapping person_id to their hands:
        {
            0: {'left_hand': [...], 'right_hand': [...]},
            1: {'left_hand': [...], 'right_hand': [...]}
        }
    """
    num_people = len(pose_landmarks_multi)

    # Initialize result structure
    person_hands = {}
    for person_idx in range(num_people):
        person_hands[person_idx] = {
            'left_hand': None,
            'right_hand': None
        }

    if not detected_hands or len(detected_hands) == 0:
        return person_hands

    # Extract wrist positions for each person
    people_wrists = []
    for person_idx, pose_landmarks in enumerate(pose_landmarks_multi):
        left_wrist, right_wrist = get_wrist_positions(pose_landmarks)
        people_wrists.append({
            'person_idx': person_idx,
            'left_wrist': left_wrist,
            'right_wrist': right_wrist
        })

    # Process each detected hand
    for hand in detected_hands:
        hand_landmarks = hand.get('landmarks', [])
        handedness = hand.get('handedness', 'Unknown')  # 'Left' or 'Right' (mirrored)

        if not hand_landmarks:
            continue

        # Get hand center position
        hand_center = get_hand_center(hand_landmarks)
        if not hand_center:
            continue

        # Find the closest person's wrist
        min_distance = float('inf')
        best_person_idx = None
        best_hand_side = None

        for wrist_info in people_wrists:
            person_idx = wrist_info['person_idx']

            # Check distance to left wrist
            if wrist_info['left_wrist']:
                dist_to_left = calculate_distance(hand_center, wrist_info['left_wrist'])
                if dist_to_left < min_distance and dist_to_left < max_distance_threshold:
                    min_distance = dist_to_left
                    best_person_idx = person_idx
                    best_hand_side = 'left_hand'

            # Check distance to right wrist
            if wrist_info['right_wrist']:
                dist_to_right = calculate_distance(hand_center, wrist_info['right_wrist'])
                if dist_to_right < min_distance and dist_to_right < max_distance_threshold:
                    min_distance = dist_to_right
                    best_person_idx = person_idx
                    best_hand_side = 'right_hand'

        # Assign hand to the best matching person
        if best_person_idx is not None and best_hand_side is not None:
            # Only assign if this person doesn't already have a hand on this side
            if person_hands[best_person_idx][best_hand_side] is None:
                person_hands[best_person_idx][best_hand_side] = hand_landmarks

    return person_hands


def match_hands_to_people_from_results(
    hand_results: any,
    pose_landmarks_multi: List[List[Dict]],
    hand_detector: any = None
) -> Dict[int, Dict[str, Optional[List[Dict]]]]:
    """
    Convenience function to match hands from MediaPipe detection results

    Args:
        hand_results: MediaPipe hand detection results object
        pose_landmarks_multi: List of pose landmarks for each person
        hand_detector: Hand detector instance (to access convert_to_dict method)

    Returns:
        Dictionary mapping person_id to their hands
    """
    detected_hands = []

    if hand_results and hasattr(hand_results, 'hand_landmarks'):
        for hand_idx, landmarks in enumerate(hand_results.hand_landmarks):
            # Convert landmarks to dict
            if hand_detector and hasattr(hand_detector, 'convert_to_dict'):
                # Create a mock result object with single hand
                class SingleHandResult:
                    def __init__(self, landmarks, world_landmarks, handedness):
                        self.hand_landmarks = [landmarks]
                        self.hand_world_landmarks = [world_landmarks] if world_landmarks else None
                        self.handedness = [handedness] if handedness else None

                # Get world landmarks and handedness for this hand
                world_landmarks = hand_results.hand_world_landmarks[hand_idx] if hand_results.hand_world_landmarks else None
                handedness = hand_results.handedness[hand_idx] if hand_results.handedness else None

                single_result = SingleHandResult(landmarks, world_landmarks, handedness)
                landmarks_dict = hand_detector.convert_to_dict(single_result)

                # Get handedness label
                # MediaPipe Tasks API: handedness is already a list of Category objects
                handedness_label = 'Unknown'
                confidence = 1.0
                if handedness and len(handedness) > 0:
                    # handedness is already a list: [Category]
                    # Handle different handedness data structures
                    first_hand = handedness[0]

                    # If it's an object with attributes
                    if hasattr(first_hand, 'category_name') and hasattr(first_hand, 'score'):
                        handedness_label = first_hand.category_name
                        confidence = first_hand.score
                    # If it's a dictionary
                    elif isinstance(first_hand, dict):
                        handedness_label = first_hand.get('category_name', first_hand.get('label', 'Unknown'))
                        confidence = first_hand.get('score', first_hand.get('confidence', 1.0))
                    # If it's a string (direct label)
                    elif isinstance(first_hand, str):
                        handedness_label = first_hand
                        confidence = 1.0

                detected_hands.append({
                    'landmarks': landmarks_dict,
                    'handedness': handedness_label,
                    'confidence': confidence
                })

    return assign_hands_to_people(detected_hands, pose_landmarks_multi)


def get_person_hand_landmarks(
    person_id: int,
    matched_hands: Dict[int, Dict[str, Optional[List[Dict]]]]
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Get left and right hand landmarks for a specific person

    Args:
        person_id: The person identifier
        matched_hands: Result from assign_hands_to_people or match_hands_to_people_from_results

    Returns:
        Tuple of (left_hand_landmarks, right_hand_landmarks)
    """
    if person_id not in matched_hands:
        return None, None

    return matched_hands[person_id]['left_hand'], matched_hands[person_id]['right_hand']


def get_nose_position(pose_landmarks: List[Dict]) -> Optional[Dict]:
    """
    Extract nose position from pose landmarks

    MediaPipe Pose landmark index:
    - 0: Nose

    Args:
        pose_landmarks: List of pose landmark dictionaries

    Returns:
        Dictionary with 'x', 'y' coordinates of nose
    """
    if pose_landmarks and len(pose_landmarks) > 0:
        nose = pose_landmarks[0]
        # Ensure nose is a dictionary before calling .get()
        if isinstance(nose, dict):
            return {'x': nose.get('x', 0), 'y': nose.get('y', 0)}
        elif hasattr(nose, 'x') and hasattr(nose, 'y'):
            # Handle case where it's an object with x, y attributes
            return {'x': nose.x, 'y': nose.y}
    return None


def get_face_center(face_landmarks: List[Dict]) -> Optional[Dict]:
    """
    Calculate face center from face mesh landmarks

    Uses nose tip (landmark 1) as the reference point

    Args:
        face_landmarks: List of face landmark dictionaries

    Returns:
        Dictionary with 'x', 'y' coordinates of face center
    """
    if not face_landmarks or len(face_landmarks) < 2:
        return None

    # MediaPipe FaceMesh: landmark 1 is the nose tip
    if len(face_landmarks) > 1:
        nose_tip = face_landmarks[1]
        # Ensure nose_tip is a dictionary before calling .get()
        if isinstance(nose_tip, dict):
            return {'x': nose_tip.get('x', 0), 'y': nose_tip.get('y', 0)}
        elif hasattr(nose_tip, 'x') and hasattr(nose_tip, 'y'):
            # Handle case where it's an object with x, y attributes
            return {'x': nose_tip.x, 'y': nose_tip.y}

    return None


def assign_faces_to_people(
    face_bboxes: List[Dict],
    pose_landmarks_multi: List[List[Dict]],
    max_distance_threshold: float = 0.3
) -> Dict[int, Optional[Dict]]:
    """
    Assign detected face bounding boxes to specific people based on spatial proximity

    Args:
        face_bboxes: List of YOLO face bounding boxes with 'x', 'y', 'w', 'h' keys
        pose_landmarks_multi: List of pose landmarks for each person
        max_distance_threshold: Maximum distance to consider a face match

    Returns:
        Dictionary mapping person_id to their face bbox:
        {
            0: {'x': ..., 'y': ..., 'w': ..., 'h': ...},
            1: {'x': ..., 'y': ..., 'w': ..., 'h': ...}
        }
    """
    num_people = len(pose_landmarks_multi)

    # Initialize result structure
    person_faces = {}
    for person_idx in range(num_people):
        person_faces[person_idx] = None

    if not face_bboxes or len(face_bboxes) == 0:
        return person_faces

    # Extract nose positions for each person
    people_noses = []
    for person_idx, pose_landmarks in enumerate(pose_landmarks_multi):
        nose_pos = get_nose_position(pose_landmarks)
        if nose_pos:
            people_noses.append({
                'person_idx': person_idx,
                'nose': nose_pos
            })

    # Match each face bbox to closest person's nose
    for bbox in face_bboxes:
        # Skip if bbox is not a dictionary
        if not isinstance(bbox, dict):
            continue

        # Calculate face center from bbox
        face_center = {
            'x': bbox.get('x', 0) + bbox.get('w', 0) / 2,
            'y': bbox.get('y', 0) + bbox.get('h', 0) / 2
        }

        # Find closest person
        min_distance = float('inf')
        best_person_idx = None

        for nose_info in people_noses:
            person_idx = nose_info['person_idx']
            nose_pos = nose_info['nose']

            dist = calculate_distance(face_center, nose_pos)
            if dist < min_distance and dist < max_distance_threshold:
                min_distance = dist
                best_person_idx = person_idx

        # Assign face to best matching person
        if best_person_idx is not None:
            if person_faces[best_person_idx] is None:  # Only assign once per person
                person_faces[best_person_idx] = bbox

    return person_faces


# Example usage:
"""
# In detect_v2_3d.py:

from src.utils.hand_person_matcher import (
    match_hands_to_people_from_results,
    get_person_hand_landmarks,
    assign_faces_to_people
)

# After detection:
pose_landmarks_multi = results.get('pose_landmarks_3d_dict', [])
hand_results = results.get('hand_results')
face_bboxes = results.get('face_bboxes')

# Match hands to people
matched_hands = match_hands_to_people_from_results(
    hand_results,
    pose_landmarks_multi,
    self.hand_detector
)

# Match faces to people
matched_faces = assign_faces_to_people(
    face_bboxes,
    pose_landmarks_multi
)

# Get landmarks for each person
for person_idx in range(len(pose_landmarks_multi)):
    left_hand, right_hand = get_person_hand_landmarks(person_idx, matched_hands)
    face_bbox = matched_faces.get(person_idx)

    # Save to database with person_id
    self.db.add_frame_to_batch(
        person_id=person_idx,
        left_hand_landmarks=left_hand,
        right_hand_landmarks=right_hand,
        face_bbox=face_bbox,
        ...
    )
"""
