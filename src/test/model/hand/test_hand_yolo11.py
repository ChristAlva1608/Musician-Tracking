#!/usr/bin/env python3
"""
YOLO11 Hand Detection Test Script
Print all landmarks and bounding boxes for analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO

def main():
    print("🤚 Starting YOLO11 Hand Detection Test")
    print("=" * 50)
    
    # Load the YOLO model
    model_path = '/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/src/checkpoints/yolo11n-hand.pt'
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    print("📹 Webcam opened successfully")
    print("Press 'q' to quit, 's' to save current detection info")
    print("=" * 50)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, verbose=False)
            
            # Process results
            for i, result in enumerate(results):
                print(f"\n📊 Frame {frame_count} - Result {i}:")
                
                # Print bounding boxes
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    print(f"   📦 Found {len(boxes)} bounding boxes:")
                    
                    for j, box in enumerate(boxes):
                        # Get box coordinates (xyxy format)
                        coords = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Get class info if available
                        if hasattr(box, 'cls') and box.cls is not None:
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names.get(class_id, f"class_{class_id}")
                        else:
                            class_name = "unknown"
                        
                        print(f"      Box {j}: {class_name}")
                        print(f"         Coordinates: x1={coords[0]:.1f}, y1={coords[1]:.1f}, x2={coords[2]:.1f}, y2={coords[3]:.1f}")
                        print(f"         Size: w={coords[2]-coords[0]:.1f}, h={coords[3]-coords[1]:.1f}")
                        print(f"         Confidence: {confidence:.3f}")
                
                # Print keypoints/landmarks if available
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints
                    print(f"   👆 Found keypoints for {len(keypoints.data)} objects:")
                    
                    for obj_idx, obj_keypoints in enumerate(keypoints.data):
                        print(f"      Object {obj_idx} keypoints:")
                        
                        # obj_keypoints shape should be [num_keypoints, 3] (x, y, visibility)
                        kpts = obj_keypoints.cpu().numpy()
                        
                        for kpt_idx, (x, y, vis) in enumerate(kpts):
                            if vis > 0:  # Only show visible keypoints
                                print(f"         Keypoint {kpt_idx}: x={x:.1f}, y={y:.1f}, visibility={vis:.3f}")
                
                # Additional result information
                if hasattr(result, 'names'):
                    print(f"   🏷️  Available classes: {result.names}")
                
                # Show total detection count
                total_detections = 0
                if hasattr(result, 'boxes') and result.boxes is not None:
                    total_detections += len(result.boxes)
                
                print(f"   📈 Total detections in this frame: {total_detections}")
            
            # Display the frame with annotations
            annotated_frame = results[0].plot() if results else frame
            cv2.imshow('YOLO11 Hand Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save detailed info to file
                filename = f"hand_detection_frame_{frame_count}.txt"
                with open(filename, 'w') as f:
                    f.write(f"Frame {frame_count} Detection Details\n")
                    f.write("=" * 40 + "\n")
                    
                    for i, result in enumerate(results):
                        f.write(f"\nResult {i}:\n")
                        
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            f.write(f"Bounding Boxes ({len(result.boxes)}):\n")
                            for j, box in enumerate(result.boxes):
                                coords = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                f.write(f"  Box {j}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}] conf={confidence:.3f}\n")
                        
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            f.write(f"Keypoints:\n")
                            for obj_idx, obj_keypoints in enumerate(result.keypoints.data):
                                f.write(f"  Object {obj_idx}:\n")
                                kpts = obj_keypoints.cpu().numpy()
                                for kpt_idx, (x, y, vis) in enumerate(kpts):
                                    if vis > 0:
                                        f.write(f"    Keypoint {kpt_idx}: ({x:.1f}, {y:.1f}) vis={vis:.3f}\n")
                
                print(f"💾 Saved detection info to {filename}")
            
            # Print summary every 30 frames
            if frame_count % 30 == 0:
                print(f"\n🔄 Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Test completed! Processed {frame_count} frames total")

if __name__ == "__main__":
    main()