import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def create_mobilenetv2_emotion_model():
    """Create MobileNetV2 model for emotion recognition"""
    # Download pretrained MobileNetV2 weights (ImageNet)
    base_model = MobileNetV2(
        weights='imagenet',  # Automatically downloads weights
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Add custom layers for emotion recognition
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)  # 7 emotions

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def preprocess_face_image(face_image, target_size=(224, 224)):
    """Preprocess face image for MobileNetV2"""
    # Resize to target size
    face_image = cv2.resize(face_image, target_size)
    
    # Convert to RGB if needed
    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Convert to array and normalize
    face_array = img_to_array(face_image)
    face_array = face_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    face_array = np.expand_dims(face_array, axis=0)
    
    return face_array

def detect_face_mediapipe(image):
    """Detect face using MediaPipe"""
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                faces.append((x, y, w, h))
        
        return faces

def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    # Preprocess the face image
    processed_image = preprocess_face_image(face_image)
    
    # Make prediction with timing
    import time
    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)
    inference_time = time.time() - start_time
    
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    
    return EMOTIONS[emotion_idx], confidence, predictions[0], inference_time

def test_with_video(video_path):
    """Test the model with video file"""
    print("Creating MobileNetV2 emotion recognition model...")
    model = create_mobilenetv2_emotion_model()
    
    print("Model created successfully!")
    print(f"Model summary:")
    model.summary()
    
    print(f"\nTesting with video: {video_path}")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    
    frame_count = 0
    total_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_inference_time = 0
        
        # Detect faces
        faces = detect_face_mediapipe(frame)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:  # Check if face region is valid
                try:
                    # Predict emotion
                    emotion, confidence, all_predictions, inference_time = predict_emotion(model, face_roi)
                    frame_inference_time += inference_time
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion, confidence and timing
                    text = f"{emotion}: {confidence:.2f} ({inference_time*1000:.0f}ms)"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        total_inference_time += frame_inference_time
        
        # Print periodic stats
        if frame_count % 30 == 0 or len(faces) > 0:
            avg_inference = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
            print(f"Frame {frame_count}: {len(faces)} faces, {frame_inference_time*1000:.1f}ms inference, avg: {avg_inference:.1f}ms")
        
        # Add frame counter and timing info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Inference: {frame_inference_time*1000:.0f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('MobileNetV2 Emotion Recognition', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print final statistics
    avg_inference = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
    print(f"\n📊 Test Results:")
    print(f"Processed {frame_count} frames")
    print(f"Average inference time: {avg_inference:.1f}ms per frame")
    print(f"Total inference time: {total_inference_time:.1f}s")
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image(image_path):
    """Test the model with a single image"""
    print(f"Testing with image: {image_path}")
    
    # Create model
    model = create_mobilenetv2_emotion_model()
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Detect faces
    faces = detect_face_mediapipe(image)
    
    if not faces:
        print("No faces detected in the image")
        return
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size > 0:
            try:
                # Predict emotion
                emotion, confidence, all_predictions, inference_time = predict_emotion(model, face_roi)
                
                print(f"\nFace {i+1}:")
                print(f"Emotion: {emotion}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Inference time: {inference_time*1000:.1f}ms")
                print("All predictions:")
                for j, (emotion_name, pred) in enumerate(zip(EMOTIONS, all_predictions)):
                    print(f"  {emotion_name}: {pred:.3f}")
                
                # Draw rectangle and label
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{emotion}: {confidence:.2f} ({inference_time*1000:.0f}ms)"
                cv2.putText(image, text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face {i+1}: {e}")
    
    # Display result
    cv2.imshow('MobileNetV2 Emotion Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("MobileNetV2 Emotion Recognition Test")
    print("=" * 40)
    
    # Test model creation
    print("1. Testing model creation...")
    try:
        model = create_mobilenetv2_emotion_model()
        print("✓ Model created successfully!")
        print(f"✓ Model has {len(model.layers)} layers")
        print(f"✓ Input shape: {model.input_shape}")
        print(f"✓ Output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        exit(1)
    
    # Run video test automatically
    print("\n2. Running video test automatically...")
    video_path = "video/moonlight_sonata/short_video.mp4"
    print(f"Using default video: {video_path}")
    test_with_video(video_path) 