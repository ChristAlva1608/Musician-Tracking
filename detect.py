import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import cv2 
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import json
from datetime import datetime

def get_hand_landmarks(hand_model, img):
    """
    Detect hand landmarks using MediaPipe Hands model
    """
    return hand_model.process(img).multi_hand_landmarks

def get_face_landmarks(face_mesh_model, img):
    """
    Detect face landmarks using MediaPipe Face Mesh model
    """
    return face_mesh_model.process(img).multi_face_landmarks

def draw_mp_hand(img, multi_hand_landmarks):
    """
    Draw hand landmarks and connections on the image
    """
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                hand_landmarks, 
                mp.solutions.hands.HAND_CONNECTIONS
            )

def draw_mp_face_mesh(img, multi_face_landmarks):
    """
    Draw face landmarks and mesh on the image
    """
    if multi_face_landmarks:
        for face_landmarks in multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                face_landmarks, 
                mp.solutions.face_mesh.FACEMESH_TESSELATION
            )

def draw_yolo_pose(img, pose_results):
    """
    Draw YOLO pose detection results
    """
    if pose_results and pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        
        # Define COCO pose connections (17 keypoints)
        pose_connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (3, 5), (4, 6)   # Neck: ears to shoulders
        ]
        for person_keypoints in keypoints:
            # Draw keypoints
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Draw connections
            for connection in pose_connections:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints)):
                    pt1 = person_keypoints[pt1_idx]
                    pt2 = person_keypoints[pt2_idx]
                    if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                        cv2.line(img, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

####### Functions to detect bad gestures #######

#--------- Turtle neck -----------#
def turtle_neck(img, pose_results):
    detected = False
    if pose_results and pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
        
        # Get key points
        left_shoulder = keypoints[5]   # [x, y]
        right_shoulder = keypoints[6]  # [x, y]
        left_ear = keypoints[3]        # [x, y] 
        right_ear = keypoints[4]       # [x, y]
        left_hip = keypoints[11]       # [x, y]
        right_hip = keypoints[12]      # [x, y]

        # Calculate centers
        base_of_the_neck = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ])
        
        ear_center = np.array([
            (left_ear[0] + right_ear[0]) / 2,
            (left_ear[1] + right_ear[1]) / 2
        ])
        
        hip_center = np.array([
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        ])

        # Calculate vectors
        neck_vector = ear_center - base_of_the_neck
        spine_vector = hip_center - base_of_the_neck
        
        # Calculate angle between vectors
        # angle = arccos(dot_product / (magnitude1 * magnitude2))
        dot_product = np.dot(neck_vector, spine_vector)
        magnitude_neck = np.linalg.norm(neck_vector)
        magnitude_spine = np.linalg.norm(spine_vector)
        
        if magnitude_neck > 0 and magnitude_spine > 0:
            cos_angle = dot_product / (magnitude_neck * magnitude_spine)
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # Check if angle < 150 degrees (turtle neck condition)
            if angle_degrees < 150:
                detected = True
                cv2.putText(img, f"TURTLE NECK {angle_degrees:.1f}°", 
                           (int(base_of_the_neck[0] - 60), int(base_of_the_neck[1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (int(base_of_the_neck[0] - 80), int(base_of_the_neck[1] - 25)), 
                             (int(base_of_the_neck[0] + 80), int(base_of_the_neck[1] + 5)), (0, 0, 255), 1)
    return detected

#--------- Hunched back -----------#
def hunched_back(img, hand_landmarker_result):
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.y < 0.5:
                    return True
    return False

#--------- Raised shoulders -----------#
def raised_shoulders(img, hand_landmarker_result):
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.y > 0.5:
                    return True
    return False

#--------- Low wrists -----------#
def low_wrists(img, multi_hand_landmarks):
    detected = False
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            # Check if wrist (landmark 0) is lower than knuckle (landmark 9) 
            wrist = hand_landmarks.landmark[0]
            middle_knuckle = hand_landmarks.landmark[9]
            if wrist.y > middle_knuckle.y:  # Higher y value means lower position
                detected = True
                # Convert normalized coordinates to pixel coordinates
                wrist_x = int(wrist.x * img.shape[1])
                wrist_y = int(wrist.y * img.shape[0])
                
                # Display alarm text at wrist location
                cv2.putText(img, "LOW-WRIST", 
                            (wrist_x - 40, wrist_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Add warning background rectangle around text
                cv2.rectangle(img, (wrist_x - 45, wrist_y - 25), 
                                (wrist_x + 55, wrist_y + 5), (0, 0, 255), 1)
    return detected

#--------- Fingers pointing up to the sky -----------#
def fingers_pointing_up_to_the_sky(img, multi_hand_landmarks):
    detected = False
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            pinky_tip = hand_landmarks.landmark[20]
            pinky_mcp = hand_landmarks.landmark[17]
            if pinky_tip.y < pinky_mcp.y:
                detected = True
                cv2.putText(img, "FINGERS POINTING UP TO THE SKY", 
                            (int(pinky_tip.x * img.shape[1]), int(pinky_tip.y * img.shape[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (int(pinky_tip.x * img.shape[1] - 20), int(pinky_tip.y * img.shape[0] - 20)), 
                             (int(pinky_tip.x * img.shape[1] + 20), int(pinky_tip.y * img.shape[0] + 20)), (0, 0, 255), 1)
    return detected


class LandmarkHeatmapGenerator:
    def __init__(self, frame_width, frame_height):
        """
        Initialize heatmap generator
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hand_heatmap_data = defaultdict(list)  # {landmark_index: [(x, y), ...]}
        self.pose_heatmap_data = defaultdict(list)   # {landmark_index: [(x, y), ...]}
        self.face_heatmap_data = defaultdict(list)   # {landmark_index: [(x, y), ...]}
        
    def add_hand_landmark_data(self, multi_hand_landmarks, landmark_index):
        """
        Add hand landmark position to heatmap data
        Args:
            multi_hand_landmarks: MediaPipe hand landmarks
            landmark_index: Index of the landmark to track (0-20)
        """
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                if landmark_index < len(hand_landmarks.landmark):
                    landmark = hand_landmarks.landmark[landmark_index]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    self.hand_heatmap_data[landmark_index].append((x, y))
    
    def add_pose_landmark_data(self, pose_results, landmark_index):
        """
        Add pose landmark position to heatmap data
        Args:
            pose_results: YOLO pose detection results
            landmark_index: Index of the landmark to track (0-16 for COCO)
        """
        if pose_results and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()
            for person_keypoints in keypoints:
                if landmark_index < len(person_keypoints):
                    x, y = person_keypoints[landmark_index]
                    if x > 0 and y > 0:  # Valid keypoint
                        self.pose_heatmap_data[landmark_index].append((int(x), int(y)))
    
    def add_face_landmark_data(self, multi_face_landmarks, landmark_index):
        """
        Add face landmark position to heatmap data
        Args:
            multi_face_landmarks: MediaPipe face landmarks
            landmark_index: Index of the landmark to track (0-467 for face mesh)
        """
        if multi_face_landmarks:
            for face_landmarks in multi_face_landmarks:
                if landmark_index < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[landmark_index]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    self.face_heatmap_data[landmark_index].append((x, y))
    
    def generate_heatmap(self, landmark_type, landmark_index, save_path=None):
        """
        Generate and display heatmap for a specific landmark
        Args:
            landmark_type: 'hand', 'pose', or 'face'
            landmark_index: Index of the landmark
            save_path: Optional path to save the heatmap image
        """
        # Select appropriate data
        if landmark_type == 'hand':
            data = self.hand_heatmap_data[landmark_index]
            title = f"Hand Landmark {landmark_index} Heatmap"
        elif landmark_type == 'pose':
            data = self.pose_heatmap_data[landmark_index]
            title = f"Pose Landmark {landmark_index} Heatmap"
        elif landmark_type == 'face':
            data = self.face_heatmap_data[landmark_index]
            title = f"Face Landmark {landmark_index} Heatmap"
        else:
            raise ValueError("landmark_type must be 'hand', 'pose', or 'face'")
        
        if not data:
            print(f"No data collected for {landmark_type} landmark {landmark_index}")
            return None
        
        # Create heatmap
        heatmap = np.zeros((self.frame_height, self.frame_width))
        
        # Add Gaussian blur for each point
        for x, y in data:
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                # Create small gaussian around each point
                y_min = max(0, y - 10)
                y_max = min(self.frame_height, y + 11)
                x_min = max(0, x - 10)
                x_max = min(self.frame_width, x + 11)
                
                for dy in range(y_min, y_max):
                    for dx in range(x_min, x_max):
                        distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                        if distance <= 10:
                            intensity = np.exp(-(distance**2) / (2 * 5**2))
                            heatmap[dy, dx] += intensity
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='bilinear', origin='upper')
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
        return heatmap
    
    def get_heatmap_overlay(self, landmark_type, landmark_index, alpha=0.6):
        """
        Get heatmap as overlay for video frames
        Args:
            landmark_type: 'hand', 'pose', or 'face'
            landmark_index: Index of the landmark
            alpha: Transparency of the overlay
        Returns:
            Colored heatmap overlay as BGR image
        """
        # Select appropriate data
        if landmark_type == 'hand':
            data = self.hand_heatmap_data[landmark_index]
        elif landmark_type == 'pose':
            data = self.pose_heatmap_data[landmark_index]
        elif landmark_type == 'face':
            data = self.face_heatmap_data[landmark_index]
        else:
            return None
        
        if not data:
            return None
        
        # Create heatmap
        heatmap = np.zeros((self.frame_height, self.frame_width))
        
        for x, y in data:
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                y_min = max(0, y - 10)
                y_max = min(self.frame_height, y + 11)
                x_min = max(0, x - 10)
                x_max = min(self.frame_width, x + 11)
                
                for dy in range(y_min, y_max):
                    for dx in range(x_min, x_max):
                        distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                        if distance <= 10:
                            intensity = np.exp(-(distance**2) / (2 * 5**2))
                            heatmap[dy, dx] += intensity
        
        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color using OpenCV colormap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_color

class GestureTimer:
    def __init__(self):
        """Initialize gesture timer for tracking bad posture durations"""
        self.gesture_start_times = {
            'low_wrists': None,
            'turtle_neck': None,
            'fingers_pointing_up': None
        }
        self.gesture_durations = {
            'low_wrists': 0,
            'turtle_neck': 0,
            'fingers_pointing_up': 0
        }
        self.total_frames = 0
        self.session_start_time = time.time()
        self.last_report_time = time.time()
        
    def start_gesture(self, gesture_type):
        """Start timing a bad gesture"""
        if self.gesture_start_times[gesture_type] is None:
            self.gesture_start_times[gesture_type] = time.time()
    
    def stop_gesture(self, gesture_type):
        """Stop timing a bad gesture and add to total duration"""
        if self.gesture_start_times[gesture_type] is not None:
            duration = time.time() - self.gesture_start_times[gesture_type]
            self.gesture_durations[gesture_type] += duration
            self.gesture_start_times[gesture_type] = None
    
    def update_frame(self):
        """Update frame count and check if report is due"""
        self.total_frames += 1
        current_time = time.time()
        
        # Check if 5 minutes have passed since last report
        if current_time - self.last_report_time >= 30:  # 300 seconds = 5 minutes
            self.generate_analysis_report()
            self.last_report_time = current_time
    
    def get_current_percentages(self):
        """Get current percentages of bad gesture time"""
        total_time = time.time() - self.session_start_time
        if total_time == 0:
            return {gesture: 0.0 for gesture in self.gesture_durations.keys()}
        
        percentages = {}
        for gesture, duration in self.gesture_durations.items():
            percentages[gesture] = (duration / total_time) * 100
        
        return percentages
    
    def generate_analysis_report(self):
        """Generate and save analysis report"""
        # Create analysis_report directory if it doesn't exist
        os.makedirs('analysis_report', exist_ok=True)
        
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report/posture_analysis_{timestamp}.json"
        
        # Calculate statistics
        total_time = time.time() - self.session_start_time
        percentages = self.get_current_percentages()
        
        report_data = {
            "timestamp": current_time.isoformat(),
            "session_duration_seconds": total_time,
            "total_frames_processed": self.total_frames,
            "gesture_durations_seconds": self.gesture_durations.copy(),
            "gesture_percentages": percentages,
            "average_bad_posture_percentage": sum(percentages.values()),
            "recommendations": self.generate_recommendations(percentages)
        }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Also save a human-readable summary
        summary_filename = f"analysis_report/posture_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("POSTURE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Duration: {total_time/60:.1f} minutes\n")
            f.write(f"Total Frames: {self.total_frames}\n\n")
            
            f.write("BAD POSTURE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for gesture, percentage in percentages.items():
                duration = self.gesture_durations[gesture]
                f.write(f"{gesture.replace('_', ' ').title()}: {percentage:.1f}% ({duration:.1f}s)\n")
            
            f.write(f"\nOverall Bad Posture: {sum(percentages.values()):.1f}%\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for rec in report_data["recommendations"]:
                f.write(f"• {rec}\n")
        
        print(f"Analysis report generated: {filename}")
        print(f"Summary saved: {summary_filename}")
    
    def generate_recommendations(self, percentages):
        """Generate recommendations based on posture percentages"""
        recommendations = []
        
        if percentages['low_wrists'] > 20:
            recommendations.append("Keep wrists elevated - consider wrist support or ergonomic setup")
        
        if percentages['turtle_neck'] > 15:
            recommendations.append("Improve neck posture - keep head aligned with spine")
        
        if percentages['fingers_pointing_up'] > 10:
            recommendations.append("Maintain natural hand position - avoid upward finger pointing")
        
        if sum(percentages.values()) > 30:
            recommendations.append("Overall posture needs improvement - take regular breaks and stretch")
        
        if not recommendations:
            recommendations.append("Great posture! Keep up the good work")
        
        return recommendations

def create_landmark_heatmap_video(video_path, landmark_type, landmark_index, output_path=None):
    """
    Example function to create a heatmap for a specific landmark across a whole video
    
    Args:
        video_path: Path to input video file
        landmark_type: 'hand', 'pose', or 'face'
        landmark_index: Index of the landmark to track
        output_path: Optional path to save heatmap image
    
    Example usage:
        # Track wrist movement (hand landmark 0)
        create_landmark_heatmap_video("video.mp4", "hand", 0, "wrist_heatmap.png")
        
        # Track nose movement (pose landmark 0)
        create_landmark_heatmap_video("video.mp4", "pose", 0, "nose_heatmap.png")
        
        # Track specific face point (face landmark 1)
        create_landmark_heatmap_video("video.mp4", "face", 1, "face_heatmap.png")
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize models
    hand_model = mp.solutions.hands.Hands()
    face_mesh_model = mp.solutions.face_mesh.FaceMesh()
    pose_model = YOLO('/Volumes/Extreme_Pro/Musician-Detection-Project/trained_models/yolo11n-pose.pt')
    
    # Initialize heatmap generator
    heatmap_gen = LandmarkHeatmapGenerator(frame_width, frame_height)
    
    print(f"Processing video to track {landmark_type} landmark {landmark_index}...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get detections based on landmark type
        if landmark_type == 'hand':
            multi_hand_landmarks = hand_model.process(img_rgb).multi_hand_landmarks
            heatmap_gen.add_hand_landmark_data(multi_hand_landmarks, landmark_index)
        elif landmark_type == 'pose':
            pose_results = pose_model(frame, verbose=False)
            heatmap_gen.add_pose_landmark_data(pose_results, landmark_index)
        elif landmark_type == 'face':
            multi_face_landmarks = face_mesh_model.process(img_rgb).multi_face_landmarks
            heatmap_gen.add_face_landmark_data(multi_face_landmarks, landmark_index)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Finished processing {frame_count} frames.")
    
    # Generate and save heatmap
    heatmap_gen.generate_heatmap(landmark_type, landmark_index, output_path)
    
    return heatmap_gen

def main():
    # Open video file or webcam
    # cap = cv2.VideoCapture("koto_instrument_1.mp4")
    cap = cv2.VideoCapture(0)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize MediaPipe models
    hand_model = mp.solutions.hands.Hands()
    face_mesh_model = mp.solutions.face_mesh.FaceMesh()
    pose_model = YOLO('/Volumes/Extreme_Pro/Musician-Detection-Project/trained_models/yolo11n-pose.pt')  # YOLO for pose detection
    
    # Initialize gesture timer
    gesture_timer = GestureTimer()
    
    frame_count = 0
    
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for MediaPipe hands
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection with MediaPipe Vision Task
        multi_hand_landmarks = get_hand_landmarks(hand_model, img_rgb)

        # Face detection with MediaPipe Vision Task (Face Mesh)
        multi_face_landmarks = get_face_landmarks(face_mesh_model, img_rgb)

        # Pose detection with YOLO
        pose_results = pose_model(frame, verbose=False)

        # Draw all detections
        draw_mp_hand(frame, multi_hand_landmarks)
        draw_yolo_pose(frame, pose_results)
        draw_mp_face_mesh(frame, multi_face_landmarks)

        # Check for bad gestures and update timer
        low_wrists_detected = low_wrists(frame, multi_hand_landmarks)
        turtle_neck_detected = turtle_neck(frame, pose_results)
        fingers_up_detected = fingers_pointing_up_to_the_sky(frame, multi_hand_landmarks)
        
        # Update gesture timer
        if low_wrists_detected:
            gesture_timer.start_gesture('low_wrists')
        else:
            gesture_timer.stop_gesture('low_wrists')
            
        if turtle_neck_detected:
            gesture_timer.start_gesture('turtle_neck')
        else:
            gesture_timer.stop_gesture('turtle_neck')
            
        if fingers_up_detected:
            gesture_timer.start_gesture('fingers_pointing_up')
        else:
            gesture_timer.stop_gesture('fingers_pointing_up')
        
        # Update frame and check for report generation
        gesture_timer.update_frame()
        # Display frame count and time
        cv2.putText(frame, f"Frame: {frame_count} | Time: {frame_count/fps:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current posture statistics
        percentages = gesture_timer.get_current_percentages()
        y_offset = 60
        cv2.putText(frame, f"Low Wrists: {percentages['low_wrists']:.1f}%", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Turtle Neck: {percentages['turtle_neck']:.1f}%", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Fingers Up: {percentages['fingers_pointing_up']:.1f}%", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Bad Posture: {sum(percentages.values()):.1f}%", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow("Multi-Model Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # Generate final analysis report
    print("Generating final analysis report...")
    gesture_timer.generate_analysis_report()
    
    print(f"Processed {frame_count} frames.")
    print("Session completed. Check the 'analysis_report' folder for detailed reports.")

if __name__ == '__main__':
    main()