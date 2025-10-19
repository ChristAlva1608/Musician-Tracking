# MediaPipe World Coordinates Guide

A comprehensive guide to understanding and working with MediaPipe 3D world landmarks.

---

## Table of Contents
1. [Understanding World Coordinates](#understanding-world-coordinates)
2. [Coordinate System Details](#coordinate-system-details)
3. [Handling Missing Hips (Low Visibility)](#handling-missing-hips)
4. [Exporting to CSV with Center Coordinates](#exporting-to-csv)
5. [Rotation Data from MediaPipe](#rotation-data-from-mediapipe)
6. [Code Examples](#code-examples)

---

## Understanding World Coordinates

### What Are World Landmarks?

MediaPipe provides **TWO types** of pose landmark outputs:

| Output Type | Coordinates | Units | Use Case |
|-------------|-------------|-------|----------|
| **`pose_landmarks`** | 2D normalized (x, y) + relative z | 0.0-1.0 (normalized) | Screen visualization |
| **`pose_world_landmarks`** | 3D real-world (x, y, z) | **METERS** | 3D analysis, measurements |

### MediaPipe Pose Landmark Indices

```
0-10:   Head (nose, eyes, ears, mouth)
11-16:  Upper body (shoulders, elbows, wrists)
17-22:  Hands (pinky, index, thumb landmarks)
23-24:  HIPS (LEFT_HIP=23, RIGHT_HIP=24) ← Origin of world coordinates!
25-28:  Legs (knees, ankles)
29-32:  Feet (heel, foot index)
```

**Key Landmarks:**
- **Landmark 23**: `LEFT_HIP`
- **Landmark 24**: `RIGHT_HIP`
- **Hip Center**: Origin (0, 0, 0) for world coordinates = midpoint of landmarks 23 and 24

---

## Coordinate System Details

### World Coordinate System

```
Origin: Midpoint between LEFT_HIP (23) and RIGHT_HIP (24)
Units:  METERS (real-world metric)

       Y (up)
       |
       |
       |______ X (right)
      /
     /
    Z (forward toward camera)
```

**Axis Orientation:**
- **X-axis**: Left (-) to Right (+)
- **Y-axis**: Down (-) to Up (+)
- **Z-axis**: Back (-) to Front/Camera (+)

### Example World Coordinates

```json
{
  "LEFT_SHOULDER": {
    "x": -0.15,      // 15cm to the left of hip center
    "y": 0.45,       // 45cm above hip center
    "z": -0.05,      // 5cm behind hip center
    "confidence": 0.95
  },
  "RIGHT_WRIST": {
    "x": 0.21,       // 21cm to the right
    "y": 0.30,       // 30cm above hips
    "z": -0.10,      // 10cm behind hips
    "confidence": 0.87
  },
  "LEFT_HIP": {
    "x": -0.08,      // ~8cm left of center (half of hip width)
    "y": 0.0,        // At origin height
    "z": 0.0,        // At origin depth
    "confidence": 0.98
  },
  "RIGHT_HIP": {
    "x": 0.08,       // ~8cm right of center
    "y": 0.0,        // At origin height
    "z": 0.0,        // At origin depth
    "confidence": 0.98
  }
}
```

### Reading the Data

```python
import json

# From database query or detection results
pose_landmarks = [
    {"x": -0.15, "y": 0.45, "z": -0.05, "confidence": 0.95},  # Landmark 0
    # ... more landmarks ...
    {"x": -0.08, "y": 0.0, "z": 0.0, "confidence": 0.98},     # Landmark 23 (LEFT_HIP)
    {"x": 0.08, "y": 0.0, "z": 0.0, "confidence": 0.98},      # Landmark 24 (RIGHT_HIP)
]

# Access specific landmark
left_hip = pose_landmarks[23]
right_hip = pose_landmarks[24]

# Calculate hip center (should be near 0, 0, 0)
hip_center_x = (left_hip['x'] + right_hip['x']) / 2
hip_center_y = (left_hip['y'] + right_hip['y']) / 2
hip_center_z = (left_hip['z'] + right_hip['z']) / 2

print(f"Hip Center: ({hip_center_x:.3f}, {hip_center_y:.3f}, {hip_center_z:.3f}) meters")
# Output: Hip Center: (0.000, 0.000, 0.000) meters
```

---

## Handling Missing Hips (Low Visibility)

### What Happens When Hips Are Not Visible?

**Scenario**: Person is sitting, lower body occluded, or far from camera.

**MediaPipe Behavior**:
- MediaPipe will still output 33 landmarks
- Hip landmarks (23, 24) will have **LOW VISIBILITY/CONFIDENCE**
- World coordinates will be **estimated** (may be less accurate)
- The origin will shift based on MediaPipe's internal estimation

### Checking Landmark Visibility

```python
def is_landmark_visible(landmark, threshold=0.5):
    """
    Check if a landmark is reliably visible

    Args:
        landmark: Dict with 'confidence' key (visibility score 0.0-1.0)
        threshold: Minimum confidence to consider visible

    Returns:
        bool: True if landmark is visible enough
    """
    return landmark.get('confidence', 0.0) >= threshold


# Example usage
pose_landmarks = [...] # 33 landmarks from database

left_hip = pose_landmarks[23]
right_hip = pose_landmarks[24]

if is_landmark_visible(left_hip) and is_landmark_visible(right_hip):
    print("✅ Both hips visible - world coordinates are reliable")
    hip_center = calculate_hip_center(left_hip, right_hip)
else:
    print("⚠️ Hip visibility low - world coordinates may be less accurate")
    print(f"   LEFT_HIP confidence: {left_hip['confidence']:.2f}")
    print(f"   RIGHT_HIP confidence: {right_hip['confidence']:.2f}")

    # Option 1: Use alternative reference points
    # Use shoulders (11, 12) as fallback origin
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    if is_landmark_visible(left_shoulder) and is_landmark_visible(right_shoulder):
        print("   Using shoulders as alternative reference")
        reference_center = calculate_midpoint(left_shoulder, right_shoulder)

    # Option 2: Skip this frame
    # Option 3: Use previous frame's hip position
```

### Strategy for Missing Hips

#### Option 1: Fallback to Alternative Reference Points

```python
def get_reference_point(pose_landmarks, confidence_threshold=0.5):
    """
    Get best available reference point for coordinate system

    Priority:
    1. Hip center (landmarks 23, 24)
    2. Shoulder center (landmarks 11, 12)
    3. Nose (landmark 0)

    Returns:
        tuple: (reference_point, reference_name)
    """
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]

    # Try hips first
    if (is_landmark_visible(left_hip, confidence_threshold) and
        is_landmark_visible(right_hip, confidence_threshold)):
        center = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2,
            'z': (left_hip['z'] + right_hip['z']) / 2,
        }
        return center, "hip_center"

    # Fallback to shoulders
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    if (is_landmark_visible(left_shoulder, confidence_threshold) and
        is_landmark_visible(right_shoulder, confidence_threshold)):
        center = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
            'z': (left_shoulder['z'] + right_shoulder['z']) / 2,
        }
        return center, "shoulder_center"

    # Last resort: use nose
    nose = pose_landmarks[0]
    if is_landmark_visible(nose, confidence_threshold):
        return nose, "nose"

    return None, "no_reference"


# Usage
reference, ref_name = get_reference_point(pose_landmarks)

if reference:
    print(f"Using {ref_name} as reference: {reference}")

    # Re-center all coordinates relative to this reference
    adjusted_landmarks = []
    for landmark in pose_landmarks:
        adjusted = {
            'x': landmark['x'] - reference['x'],
            'y': landmark['y'] - reference['y'],
            'z': landmark['z'] - reference['z'],
            'confidence': landmark['confidence']
        }
        adjusted_landmarks.append(adjusted)
else:
    print("⚠️ No reliable reference point found - skip frame")
```

#### Option 2: Temporal Smoothing (Use Previous Frame)

```python
class PoseTracker:
    def __init__(self):
        self.previous_hip_center = None

    def get_stable_hip_center(self, pose_landmarks, confidence_threshold=0.5):
        """
        Get hip center with temporal smoothing
        """
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]

        if (is_landmark_visible(left_hip, confidence_threshold) and
            is_landmark_visible(right_hip, confidence_threshold)):
            # Calculate current hip center
            current = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2,
            }
            self.previous_hip_center = current
            return current

        elif self.previous_hip_center is not None:
            # Use previous frame's hip center
            print("⚠️ Using previous frame's hip center")
            return self.previous_hip_center

        else:
            # No hip data available
            return None
```

---

## Exporting to CSV with Center Coordinates

### Option 1: Export with Hip Center Column

```python
import csv
import json

def export_pose_to_csv(database_results, output_file='pose_data.csv'):
    """
    Export pose landmarks to CSV with explicit hip center coordinates

    Args:
        database_results: Query results from musician_frame_analysis table
        output_file: Path to output CSV file
    """

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'session_id', 'frame_number', 'video_file',
            'original_time', 'synced_time',
            'hip_center_x', 'hip_center_y', 'hip_center_z',
            'hip_center_confidence',
            'pose_landmarks_json'  # Full JSON for detailed analysis
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in database_results:
            # Parse pose landmarks JSON
            pose_landmarks = json.loads(row['pose_landmarks']) if row['pose_landmarks'] else []

            if len(pose_landmarks) >= 25:  # Ensure we have hip landmarks
                left_hip = pose_landmarks[23]
                right_hip = pose_landmarks[24]

                # Calculate hip center
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                hip_center_y = (left_hip['y'] + right_hip['y']) / 2
                hip_center_z = (left_hip['z'] + right_hip['z']) / 2
                hip_center_confidence = (left_hip['confidence'] + right_hip['confidence']) / 2
            else:
                hip_center_x = hip_center_y = hip_center_z = 0.0
                hip_center_confidence = 0.0

            writer.writerow({
                'session_id': row['session_id'],
                'frame_number': row['frame_number'],
                'video_file': row['video_file'],
                'original_time': row['original_time'],
                'synced_time': row['synced_time'],
                'hip_center_x': f"{hip_center_x:.6f}",
                'hip_center_y': f"{hip_center_y:.6f}",
                'hip_center_z': f"{hip_center_z:.6f}",
                'hip_center_confidence': f"{hip_center_confidence:.3f}",
                'pose_landmarks_json': json.dumps(pose_landmarks)
            })

    print(f"✅ Exported to {output_file}")


# Usage with database query
from src.database.database_setup_v2 import DatabaseManager

db = DatabaseManager()
results = db.session.query(db.MusicianFrameAnalysis).filter_by(
    session_id='session_v2_20250119_143022'
).all()

# Convert SQLAlchemy results to dict
results_dict = [
    {
        'session_id': r.session_id,
        'frame_number': r.frame_number,
        'video_file': r.video_file,
        'original_time': r.original_time,
        'synced_time': r.synced_time,
        'pose_landmarks': json.dumps(r.pose_landmarks)
    }
    for r in results
]

export_pose_to_csv(results_dict, 'pose_with_hip_center.csv')
```

### Option 2: Export All Landmarks Individually

```python
def export_detailed_pose_csv(database_results, output_file='pose_detailed.csv'):
    """
    Export pose with each landmark as separate columns
    """

    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    # Build fieldnames
    fieldnames = ['session_id', 'frame_number', 'video_file', 'synced_time']
    fieldnames += ['hip_center_x', 'hip_center_y', 'hip_center_z']

    for name in landmark_names:
        fieldnames.extend([
            f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_conf'
        ])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in database_results:
            pose_landmarks = json.loads(row['pose_landmarks']) if row['pose_landmarks'] else []

            if len(pose_landmarks) < 33:
                continue  # Skip incomplete data

            # Calculate hip center
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2,
            }

            # Build row data
            row_data = {
                'session_id': row['session_id'],
                'frame_number': row['frame_number'],
                'video_file': row['video_file'],
                'synced_time': row['synced_time'],
                'hip_center_x': f"{hip_center['x']:.6f}",
                'hip_center_y': f"{hip_center['y']:.6f}",
                'hip_center_z': f"{hip_center['z']:.6f}",
            }

            # Add all landmarks
            for i, name in enumerate(landmark_names):
                landmark = pose_landmarks[i]
                row_data[f'{name}_x'] = f"{landmark['x']:.6f}"
                row_data[f'{name}_y'] = f"{landmark['y']:.6f}"
                row_data[f'{name}_z'] = f"{landmark['z']:.6f}"
                row_data[f'{name}_conf'] = f"{landmark['confidence']:.3f}"

            writer.writerow(row_data)

    print(f"✅ Exported detailed pose to {output_file}")
```

---

## Rotation Data from MediaPipe

### Does MediaPipe Provide Rotation Data?

**Short Answer**: MediaPipe Pose **does NOT directly output rotation matrices or Euler angles**.

However, you can **calculate rotations** from the landmark positions:

### What Rotations Can You Calculate?

1. **Head Rotation** (Pitch, Yaw, Roll)
2. **Torso Rotation**
3. **Limb Joint Angles**
4. **Body Orientation**

### Calculating Head Rotation

```python
import numpy as np
import math

def calculate_head_rotation(pose_landmarks):
    """
    Calculate head rotation (pitch, yaw, roll) from face landmarks

    Returns:
        dict: {'pitch': float, 'yaw': float, 'roll': float} in degrees
    """
    # Get face landmarks (using ears and nose as reference)
    nose = np.array([
        pose_landmarks[0]['x'],
        pose_landmarks[0]['y'],
        pose_landmarks[0]['z']
    ])

    left_ear = np.array([
        pose_landmarks[7]['x'],
        pose_landmarks[7]['y'],
        pose_landmarks[7]['z']
    ])

    right_ear = np.array([
        pose_landmarks[8]['x'],
        pose_landmarks[8]['y'],
        pose_landmarks[8]['z']
    ])

    # Calculate head center
    head_center = (left_ear + right_ear) / 2

    # Calculate forward vector (from head center to nose)
    forward_vec = nose - head_center
    forward_vec = forward_vec / np.linalg.norm(forward_vec)  # Normalize

    # Calculate right vector (from left ear to right ear)
    right_vec = right_ear - left_ear
    right_vec = right_vec / np.linalg.norm(right_vec)

    # Calculate up vector (cross product)
    up_vec = np.cross(right_vec, forward_vec)
    up_vec = up_vec / np.linalg.norm(up_vec)

    # Calculate rotation angles
    # Pitch: rotation around X-axis (nodding)
    pitch = math.atan2(-forward_vec[1], forward_vec[2])

    # Yaw: rotation around Y-axis (turning left/right)
    yaw = math.atan2(forward_vec[0], forward_vec[2])

    # Roll: rotation around Z-axis (tilting)
    roll = math.atan2(right_vec[1], right_vec[0])

    return {
        'pitch': math.degrees(pitch),
        'yaw': math.degrees(yaw),
        'roll': math.degrees(roll)
    }


# Example usage
pose_landmarks = [...]  # 33 landmarks from database

head_rotation = calculate_head_rotation(pose_landmarks)
print(f"Head Pitch: {head_rotation['pitch']:.1f}°")  # Nodding up/down
print(f"Head Yaw: {head_rotation['yaw']:.1f}°")      # Turning left/right
print(f"Head Roll: {head_rotation['roll']:.1f}°")    # Tilting left/right
```

### Calculating Torso Rotation

```python
def calculate_torso_rotation(pose_landmarks):
    """
    Calculate torso rotation from shoulders and hips
    """
    # Get shoulder landmarks
    left_shoulder = np.array([
        pose_landmarks[11]['x'],
        pose_landmarks[11]['y'],
        pose_landmarks[11]['z']
    ])

    right_shoulder = np.array([
        pose_landmarks[12]['x'],
        pose_landmarks[12]['y'],
        pose_landmarks[12]['z']
    ])

    # Get hip landmarks
    left_hip = np.array([
        pose_landmarks[23]['x'],
        pose_landmarks[23]['y'],
        pose_landmarks[23]['z']
    ])

    right_hip = np.array([
        pose_landmarks[24]['x'],
        pose_landmarks[24]['y'],
        pose_landmarks[24]['z']
    ])

    # Calculate shoulder and hip centers
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    # Calculate spine vector
    spine_vec = shoulder_center - hip_center
    spine_vec = spine_vec / np.linalg.norm(spine_vec)

    # Calculate shoulder direction (left to right)
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_vec = shoulder_vec / np.linalg.norm(shoulder_vec)

    # Calculate forward direction (cross product)
    forward_vec = np.cross(shoulder_vec, spine_vec)
    forward_vec = forward_vec / np.linalg.norm(forward_vec)

    # Calculate rotation angles
    # Torso lean (pitch)
    torso_pitch = math.atan2(-spine_vec[0], spine_vec[1])

    # Torso twist (yaw)
    torso_yaw = math.atan2(forward_vec[0], forward_vec[2])

    # Torso tilt (roll)
    torso_roll = math.atan2(shoulder_vec[1], shoulder_vec[0])

    return {
        'pitch': math.degrees(torso_pitch),  # Forward/backward lean
        'yaw': math.degrees(torso_yaw),      # Twist left/right
        'roll': math.degrees(torso_roll)     # Tilt left/right
    }
```

### Export Rotations to CSV

```python
def export_pose_with_rotations(database_results, output_file='pose_with_rotations.csv'):
    """
    Export pose data with calculated rotations
    """

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'session_id', 'frame_number', 'synced_time',
            'hip_center_x', 'hip_center_y', 'hip_center_z',
            'head_pitch', 'head_yaw', 'head_roll',
            'torso_pitch', 'torso_yaw', 'torso_roll'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in database_results:
            pose_landmarks = json.loads(row['pose_landmarks']) if row['pose_landmarks'] else []

            if len(pose_landmarks) < 33:
                continue

            # Calculate hip center
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            hip_center = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2,
            }

            # Calculate rotations
            try:
                head_rot = calculate_head_rotation(pose_landmarks)
                torso_rot = calculate_torso_rotation(pose_landmarks)
            except Exception as e:
                print(f"⚠️ Error calculating rotations for frame {row['frame_number']}: {e}")
                continue

            writer.writerow({
                'session_id': row['session_id'],
                'frame_number': row['frame_number'],
                'synced_time': row['synced_time'],
                'hip_center_x': f"{hip_center['x']:.6f}",
                'hip_center_y': f"{hip_center['y']:.6f}",
                'hip_center_z': f"{hip_center['z']:.6f}",
                'head_pitch': f"{head_rot['pitch']:.2f}",
                'head_yaw': f"{head_rot['yaw']:.2f}",
                'head_roll': f"{head_rot['roll']:.2f}",
                'torso_pitch': f"{torso_rot['pitch']:.2f}",
                'torso_yaw': f"{torso_rot['yaw']:.2f}",
                'torso_roll': f"{torso_rot['roll']:.2f}",
            })

    print(f"✅ Exported pose with rotations to {output_file}")
```

---

## Code Examples

### Complete Example: Database to CSV with All Features

```python
import json
import csv
import numpy as np
import math
from src.database.database_setup_v2 import DatabaseManager

def is_landmark_visible(landmark, threshold=0.5):
    """Check if landmark is reliably visible"""
    return landmark.get('confidence', 0.0) >= threshold

def calculate_hip_center(pose_landmarks):
    """Calculate hip center from landmarks 23 and 24"""
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    return {
        'x': (left_hip['x'] + right_hip['x']) / 2,
        'y': (left_hip['y'] + right_hip['y']) / 2,
        'z': (left_hip['z'] + right_hip['z']) / 2,
        'confidence': (left_hip['confidence'] + right_hip['confidence']) / 2
    }

def export_complete_analysis(session_id, output_file='complete_pose_analysis.csv'):
    """
    Complete export with hip center, visibility checks, and rotations
    """

    # Connect to database
    db = DatabaseManager()

    # Query data
    results = db.session.query(db.MusicianFrameAnalysis).filter_by(
        session_id=session_id
    ).order_by(db.MusicianFrameAnalysis.frame_number).all()

    print(f"📊 Found {len(results)} frames for session {session_id}")

    # Export to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'frame_number', 'synced_time',
            'hip_center_x', 'hip_center_y', 'hip_center_z', 'hip_visibility',
            'reference_type',  # 'hip_center', 'shoulder_center', or 'none'
            'head_pitch', 'head_yaw', 'head_roll',
            'torso_pitch', 'torso_yaw', 'torso_roll',
            'has_bad_gestures'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Parse pose landmarks
            pose_landmarks = row.pose_landmarks

            if not pose_landmarks or len(pose_landmarks) < 33:
                print(f"⚠️ Frame {row.frame_number}: Incomplete pose data")
                continue

            # Calculate hip center
            hip_center = calculate_hip_center(pose_landmarks)

            # Determine reference type
            if hip_center['confidence'] >= 0.5:
                reference_type = 'hip_center'
            elif (is_landmark_visible(pose_landmarks[11]) and
                  is_landmark_visible(pose_landmarks[12])):
                reference_type = 'shoulder_center'
            else:
                reference_type = 'none'

            # Calculate rotations (with error handling)
            try:
                head_rot = calculate_head_rotation(pose_landmarks)
                torso_rot = calculate_torso_rotation(pose_landmarks)
            except Exception as e:
                print(f"⚠️ Frame {row.frame_number}: Rotation calculation failed")
                head_rot = {'pitch': 0, 'yaw': 0, 'roll': 0}
                torso_rot = {'pitch': 0, 'yaw': 0, 'roll': 0}

            # Check for bad gestures
            has_bad_gestures = (
                row.flag_low_wrists or
                row.flag_turtle_neck or
                row.flag_hunched_back or
                row.flag_fingers_pointing_up
            )

            writer.writerow({
                'frame_number': row.frame_number,
                'synced_time': f"{float(row.synced_time):.3f}",
                'hip_center_x': f"{hip_center['x']:.6f}",
                'hip_center_y': f"{hip_center['y']:.6f}",
                'hip_center_z': f"{hip_center['z']:.6f}",
                'hip_visibility': f"{hip_center['confidence']:.3f}",
                'reference_type': reference_type,
                'head_pitch': f"{head_rot['pitch']:.2f}",
                'head_yaw': f"{head_rot['yaw']:.2f}",
                'head_roll': f"{head_rot['roll']:.2f}",
                'torso_pitch': f"{torso_rot['pitch']:.2f}",
                'torso_yaw': f"{torso_rot['yaw']:.2f}",
                'torso_roll': f"{torso_rot['roll']:.2f}",
                'has_bad_gestures': has_bad_gestures
            })

    print(f"✅ Complete analysis exported to {output_file}")
    db.close()


# Usage
export_complete_analysis('session_v2_20250119_143022', 'my_analysis.csv')
```

---

## Summary

### Key Takeaways

1. **World Coordinates**: In METERS, origin at hip center (midpoint of landmarks 23 & 24)
2. **Missing Hips**: Check `confidence` values, use fallback reference points (shoulders, nose)
3. **CSV Export**: Always include hip center coordinates for reference
4. **Rotations**: Calculate from landmark positions (pitch, yaw, roll)
5. **MediaPipe doesn't provide**: Direct rotation matrices, but you can calculate them

### Recommended Workflow

```python
1. Query database for pose_landmarks
2. Check hip visibility (landmarks 23, 24)
3. Calculate hip center (or fallback reference)
4. Calculate rotations from landmark geometry
5. Export to CSV with:
   - Hip center coordinates
   - Rotation angles
   - Visibility/confidence scores
   - Reference type used
```

### Next Steps

- Run the export scripts on your data
- Visualize hip center and rotations over time
- Use rotation data for posture analysis
- Implement temporal smoothing for noisy data

---

**Need Help?** Check the code examples in this guide or refer to MediaPipe documentation at:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
