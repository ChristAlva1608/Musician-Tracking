#!/usr/bin/env python3
"""
Test script for MediaPipe FaceMesh model
Tests face detection and landmark drawing
"""

import cv2
import numpy as np
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.facemesh.mediapipe import MediaPipeFaceMeshDetector

def main():
    print("🧪 Testing MediaPipe FaceMesh")
    print("=" * 50)
    
    # Initialize face detector
    try:
        face_detector = MediaPipeFaceMeshDetector(min_detection_confidence=0.3)
        print("✅ MediaPipe FaceMesh initialized")
    except Exception as e:
        print(f"❌ Failed to initialize FaceMesh: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture('/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/face closeup/cam_1.mp4')
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    # Get video properties for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    output_filename = 'facemesh_output_with_annotations.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to initialize video writer")
        return
    
    print("📹 Webcam opened successfully")
    print(f"📹 Output video will be saved as: {output_filename}")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    frame_count = 0
    total_frames_with_facemesh = 0
    total_facemesh_time = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Detect face landmarks with timing
            facemesh_start = time.time()
            results = face_detector.detect(frame)
            face_landmarks = face_detector.convert_to_dict(results)
            facemesh_end = time.time()
            facemesh_time = facemesh_end - facemesh_start
            total_facemesh_time += facemesh_time
            
            # Count frames with successful facemesh detection
            if face_landmarks and len(face_landmarks) > 0:
                total_frames_with_facemesh += 1
            
            # Draw landmarks on frame
            annotated_frame = face_detector.draw_landmarks(frame, results)
            
            # Add info text
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Frames w/ FaceMesh: {total_frames_with_facemesh}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detection status
            if face_landmarks and len(face_landmarks) > 0:
                cv2.putText(annotated_frame, f"Landmarks: {len(face_landmarks)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No Face Detected", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add timing info
            if frame_count > 0:
                avg_facemesh_time = total_facemesh_time / frame_count * 1000  # Convert to ms
                cv2.putText(annotated_frame, f"FaceMesh: {avg_facemesh_time:.1f}ms avg", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Write frame to output video (with all annotations)
            out.write(annotated_frame)
            
            # Show frame
            cv2.imshow('MediaPipe FaceMesh Test', annotated_frame)
            frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
    
    # Cleanup
    cap.release()
    out.release()  # Close video writer
    cv2.destroyAllWindows()
    face_detector.cleanup()
    
    print(f"\n✅ Test completed!")
    print(f"📊 Processing Summary:")
    print(f"   • Total frames processed: {frame_count}")
    print(f"   • Total frames with successful FaceMesh: {total_frames_with_facemesh}")
    if frame_count > 0:
        print(f"   • FaceMesh success rate: {total_frames_with_facemesh / frame_count * 100:.1f}% of frames")
        print(f"   • Average FaceMesh processing time: {total_facemesh_time / frame_count * 1000:.1f}ms")
        print(f"   • Total FaceMesh time: {total_facemesh_time:.2f}s")
    print(f"📹 Output video saved as: {output_filename}")

if __name__ == "__main__":
    main()