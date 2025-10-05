"""
Better Hunchback Detection Logic

Hunchback (kyphosis) is characterized by:
1. Forward curvature of the upper spine
2. Rounded shoulders
3. Head forward position (but this is secondary)

The key is detecting SPINE CURVATURE, not just head position.
"""

import numpy as np

def proper_hunchback_detection(landmarks):
    """
    Proper hunchback detection focusing on spine curvature
    
    Key measurements:
    1. Shoulder angle (are shoulders rounded forward?)
    2. Spine curvature (is upper back curved?)
    3. Head position relative to spine (secondary)
    """
    
    # Get key landmarks
    left_shoulder = landmarks[11]   # Left shoulder
    right_shoulder = landmarks[12]  # Right shoulder
    left_ear = landmarks[7]         # Left ear
    right_ear = landmarks[8]        # Right ear
    left_hip = landmarks[23]        # Left hip
    right_hip = landmarks[24]       # Right hip
    
    # Calculate centers
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    ear_center = (left_ear + right_ear) / 2
    
    # 1. SHOULDER ANGLE (Primary indicator)
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
    shoulder_angle_degrees = np.degrees(shoulder_angle)
    
    # Rounded shoulders: shoulders rotated forward
    shoulder_rounded_threshold = 15  # degrees
    is_shoulders_rounded = abs(shoulder_angle_degrees) > shoulder_rounded_threshold
    
    # 2. SPINE CURVATURE (Primary indicator)
    spine_vector = hip_center - shoulder_center
    spine_length = np.linalg.norm(spine_vector)
    
    if spine_length > 0:
        # Calculate spine angle relative to vertical
        spine_angle = np.arctan2(spine_vector[0], spine_vector[1])  # Relative to vertical
        spine_angle_degrees = np.degrees(spine_angle)
        
        # Forward spine curvature: spine bent forward
        spine_curved_threshold = 20  # degrees
        is_spine_curved = abs(spine_angle_degrees) > spine_curved_threshold
        
        # 3. HEAD FORWARD POSITION (Secondary indicator)
        head_to_spine_vector = ear_center - shoulder_center
        
        # Project head onto spine line
        spine_unit = spine_vector / spine_length
        projection_length = np.dot(head_to_spine_vector, spine_unit)
        projected_point = shoulder_center + projection_length * spine_unit
        
        # Calculate forward distance
        forward_vector = ear_center - projected_point
        forward_distance = np.linalg.norm(forward_vector)
        
        # Head forward threshold
        head_forward_threshold = spine_length * 0.4
        is_head_forward = forward_distance > head_forward_threshold
        
        # Combined detection logic
        # Primary: spine curvature and shoulder rounding
        # Secondary: head position
        is_hunchback = (is_spine_curved and is_shoulders_rounded) or (is_head_forward and is_shoulders_rounded)
        
        return {
            'is_hunchback': is_hunchback,
            'shoulder_angle': shoulder_angle_degrees,
            'spine_angle': spine_angle_degrees,
            'head_forward_distance': forward_distance,
            'is_shoulders_rounded': is_shoulders_rounded,
            'is_spine_curved': is_spine_curved,
            'is_head_forward': is_head_forward
        }
    
    return None

def turtle_neck_detection(landmarks):
    """
    Turtle neck detection (for comparison)
    
    Focuses on head forward position relative to shoulder line
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    nose = landmarks[0]
    
    # Calculate shoulder line
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_length = np.linalg.norm(shoulder_vector)
    
    if shoulder_length > 0:
        # Project nose onto shoulder line
        nose_to_shoulder = nose - left_shoulder
        shoulder_unit = shoulder_vector / shoulder_length
        projection_length = np.dot(nose_to_shoulder, shoulder_unit)
        projected_point = left_shoulder + projection_length * shoulder_unit
        
        # Calculate forward distance
        forward_distance = np.linalg.norm(nose - projected_point)
        
        # Turtle neck threshold
        forward_threshold = shoulder_length * 0.25
        is_turtle_neck = forward_distance > forward_threshold
        
        return {
            'is_turtle_neck': is_turtle_neck,
            'forward_distance': forward_distance,
            'threshold': forward_threshold
        }
    
    return None

# Example comparison
if __name__ == "__main__":
    # Simulated landmarks (normalized coordinates)
    landmarks = {
        0: np.array([0.5, 0.2]),   # Nose
        7: np.array([0.45, 0.15]), # Left ear
        8: np.array([0.55, 0.15]), # Right ear
        11: np.array([0.4, 0.3]),  # Left shoulder
        12: np.array([0.6, 0.3]),  # Right shoulder
        23: np.array([0.45, 0.7]), # Left hip
        24: np.array([0.55, 0.7])  # Right hip
    }
    
    # Test hunchback detection
    hunchback_result = proper_hunchback_detection(landmarks)
    print("Hunchback Detection:")
    print(f"  Detected: {hunchback_result['is_hunchback']}")
    print(f"  Shoulder angle: {hunchback_result['shoulder_angle']:.1f}°")
    print(f"  Spine angle: {hunchback_result['spine_angle']:.1f}°")
    print(f"  Head forward: {hunchback_result['head_forward_distance']:.3f}")
    
    # Test turtle neck detection
    turtle_result = turtle_neck_detection(landmarks)
    print("\nTurtle Neck Detection:")
    print(f"  Detected: {turtle_result['is_turtle_neck']}")
    print(f"  Forward distance: {turtle_result['forward_distance']:.3f}")
    
    print("\nKey Difference:")
    print("- Hunchback: Focuses on spine curvature and shoulder rounding")
    print("- Turtle Neck: Focuses on head forward position") 