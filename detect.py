import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import cv2 
import numpy as np
from ultralytics import YOLO

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
def turtle_neck(img, hand_landmarker_result):
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.y > 0.5:
                    return True
    return False

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
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            # Check if wrist (landmark 0) is lower than knuckle (landmark 9) 
            wrist = hand_landmarks.landmark[0]
            middle_knuckle = hand_landmarks.landmark[9]
            if wrist.y > middle_knuckle.y:  # Higher y value means lower position
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
                # return True
    # return False

#--------- Fingers pointing up to the sky -----------#
def fingers_pointing_up_to_the_sky(img, hand_landmarker_result):
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.y > 0.5:
                    return True
    return False

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

        # Check for low wrists and display alarm
        low_wrists(frame, multi_hand_landmarks)

        # Display frame count and time
        cv2.putText(frame, f"Frame: {frame_count} | Time: {frame_count/fps:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display detection info
        cv2.putText(frame, "Hands: MediaPipe | Pose: YOLO | Face: MediaPipe", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Multi-Model Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")

if __name__ == '__main__':
    main()