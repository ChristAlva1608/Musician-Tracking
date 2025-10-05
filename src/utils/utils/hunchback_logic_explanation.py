"""
Corrected Hunchback Detection Logic

The proper approach should be:

1. Point A: Middle of shoulders (shoulder_center)
2. Point B: Middle of hips (hip_center) 
3. Point C: Head/ear center (ear_center)

Then calculate:
- Vector AB: Spine direction (shoulder_center to hip_center)
- Vector AC: Head position relative to spine (ear_center to shoulder_center)

If the angle between AB and AC is < 160 degrees, it indicates hunchback posture.

This measures how much the head is forward relative to the spine.
"""

import numpy as np

def corrected_hunchback_detection(shoulder_center, hip_center, ear_center):
    """
    Corrected hunchback detection using proper vectors
    
    Args:
        shoulder_center: [x, y] - middle of shoulders
        hip_center: [x, y] - middle of hips  
        ear_center: [x, y] - middle of ears/head
    
    Returns:
        bool: True if hunchback detected
        float: angle in degrees
    """
    # Convert to numpy arrays
    A = np.array(shoulder_center)  # Shoulder center
    B = np.array(hip_center)       # Hip center
    C = np.array(ear_center)       # Ear/head center
    
    # Calculate vectors
    AB = B - A  # Spine vector (shoulder to hip)
    AC = C - A  # Head vector (shoulder to ear)
    
    # Calculate angle between vectors
    dot_product = np.dot(AB, AC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_AC = np.linalg.norm(AC)
    
    if magnitude_AB > 0 and magnitude_AC > 0:
        cos_angle = dot_product / (magnitude_AB * magnitude_AC)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        # Hunchback detection: if angle < 160 degrees
        is_hunchback = angle_degrees < 160
        
        return is_hunchback, angle_degrees
    
    return False, 0.0

def alternative_hunchback_detection(shoulder_center, hip_center, ear_center):
    """
    Alternative approach: Measure head forward position
    
    This measures how far forward the head is relative to the spine.
    """
    A = np.array(shoulder_center)
    B = np.array(hip_center)
    C = np.array(ear_center)
    
    # Calculate spine vector
    spine_vector = B - A
    spine_length = np.linalg.norm(spine_vector)
    
    if spine_length > 0:
        # Calculate head forward distance
        head_to_spine_vector = C - A
        
        # Project head position onto spine line
        spine_unit = spine_vector / spine_length
        projection_length = np.dot(head_to_spine_vector, spine_unit)
        projected_point = A + projection_length * spine_unit
        
        # Calculate forward distance (perpendicular to spine)
        forward_vector = C - projected_point
        forward_distance = np.linalg.norm(forward_vector)
        
        # Hunchback threshold: head forward by >30% of spine length
        forward_threshold = spine_length * 0.3
        is_hunchback = forward_distance > forward_threshold
        
        return is_hunchback, forward_distance, spine_length
    
    return False, 0.0, 0.0

# Example usage
if __name__ == "__main__":
    # Example coordinates (normalized)
    shoulder_center = [0.5, 0.3]  # Middle of shoulders
    hip_center = [0.5, 0.7]       # Middle of hips
    ear_center = [0.6, 0.2]       # Head forward and up
    
    # Test corrected logic
    is_hunchback, angle = corrected_hunchback_detection(shoulder_center, hip_center, ear_center)
    print(f"Corrected Logic - Hunchback: {is_hunchback}, Angle: {angle:.1f}°")
    
    # Test alternative logic
    is_hunchback_alt, forward_dist, spine_len = alternative_hunchback_detection(shoulder_center, hip_center, ear_center)
    print(f"Alternative Logic - Hunchback: {is_hunchback_alt}, Forward: {forward_dist:.3f}, Spine: {spine_len:.3f}") 