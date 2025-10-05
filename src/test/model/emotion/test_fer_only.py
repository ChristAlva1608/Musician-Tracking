import cv2
import numpy as np
from fer import FER
import time

def test_fer_model():
    """
    Test FER emotion detection model with webcam
    """
    print("Initializing FER model...")
    
    # Initialize the FER emotion detector
    detector = FER(mtcnn=True)
    print("✅ FER model loaded successfully")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    total_inference_time = 0
    
    while True:
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Detect emotions in the current frame with timing
        inference_start = time.time()
        emotions_in_frame = detector.detect_emotions(frame)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # Process and display results
        for emotion_info in emotions_in_frame:
            (x, y, w, h) = emotion_info["box"]
            top_emotion = max(emotion_info["emotions"], key=emotion_info["emotions"].get)
            emotion_score = emotion_info["emotions"][top_emotion]
            
            # Draw green box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with the top emotion and its score
            text = f"{top_emotion}: {emotion_score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display all emotions for this face
            y_offset = 20
            for emotion, score in emotion_info["emotions"].items():
                if score > 0.1:  # Only show emotions with >10% confidence
                    emotion_text = f"{emotion}: {score:.2f}"
                    cv2.putText(frame, emotion_text, (x, y + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 15
        
        # Calculate timing and display performance stats
        frame_count += 1
        frame_time = (time.time() - frame_start_time) * 1000  # Total frame time in ms
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference_time = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
        
        # Display performance stats
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(emotions_in_frame)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time*1000:.0f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Avg Inference: {avg_inference_time:.0f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame Time: {frame_time:.0f}ms", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Print periodic stats to console
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Inference={inference_time*1000:.0f}ms, Avg={avg_inference_time:.0f}ms, FPS={fps:.1f}")
        
        # Show the frame
        cv2.imshow("FER Emotion Detection Test", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\n📊 Test Results:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Test duration: {elapsed_time:.1f} seconds")
    print(f"Total inference time: {total_inference_time:.1f} seconds")
    print(f"Average inference time per frame: {avg_inference_time:.0f}ms")
    print(f"Inference overhead: {(total_inference_time/elapsed_time)*100:.1f}% of total time")
    
    # Performance analysis
    if avg_inference_time > 100:
        print(f"⚠️ FER inference is slower than expected ({avg_inference_time:.0f}ms)")
    elif avg_inference_time > 50:
        print(f"⚠️ FER inference is moderate ({avg_inference_time:.0f}ms)")
    else:
        print(f"✅ FER inference is fast ({avg_inference_time:.0f}ms)")

if __name__ == "__main__":
    test_fer_model() 