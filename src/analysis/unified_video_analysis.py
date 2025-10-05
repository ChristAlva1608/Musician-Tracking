#!/usr/bin/env python3
"""
Unified Video Emotion Analysis
Supports multiple emotion recognition models: DeepFace, GhostFaceNetsV2, and FER MTCNN
"""

import cv2
import numpy as np
import os
import time
from datetime import timedelta
from collections import defaultdict

# Try to import required libraries
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    import mediapipe as mp
    from ellzaf_ml.models import GhostFaceNetsV2
    from ellzaf_ml.tools.load_pretrained import load_pretrained
    GHOSTFACENETS_AVAILABLE = True
except ImportError:
    GHOSTFACENETS_AVAILABLE = False

# Import chart functions from video_emotion_analysis
from video_emotion_analysis import create_emotion_chart, get_available_checkpoints

# Model configurations
MODEL_CONFIGS = {
    'deepface': {
        'name': 'DeepFace',
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'colors': {
            'angry': '#FF0000',      # Red
            'disgust': '#8B4513',    # Brown
            'fear': '#800080',       # Purple
            'happy': '#00FF00',      # Green
            'sad': '#0000FF',        # Blue
            'surprise': '#FFFF00',   # Yellow
            'neutral': '#808080'     # Gray
        },
        'available': DEEPFACE_AVAILABLE,
        'install_cmd': 'pip install deepface'
    },
    'ghostfacenetsv2': {
        'name': 'GhostFaceNetsV2',
        'emotions': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        'colors': {
            'Angry': '#FF0000',      # Red
            'Disgust': '#8B4513',    # Brown
            'Fear': '#800080',       # Purple
            'Happy': '#00FF00',      # Green
            'Sad': '#0000FF',        # Blue
            'Surprise': '#FFFF00',   # Yellow
            'Neutral': '#808080'     # Gray
        },
        'available': GHOSTFACENETS_AVAILABLE,
        'install_cmd': 'pip install torch mediapipe ellzaf-ml'
    },
    'fer_mtcnn': {
        'name': 'FER MTCNN',
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'colors': {
            'angry': '#FF0000',      # Red
            'disgust': '#8B4513',    # Brown
            'fear': '#800080',       # Purple
            'happy': '#00FF00',      # Green
            'sad': '#0000FF',        # Blue
            'surprise': '#FFFF00',   # Yellow
            'neutral': '#808080'     # Gray
        },
        'available': FER_AVAILABLE,
        'install_cmd': 'pip install fer'
    }
}

class UnifiedVideoAnalyzer:
    """Unified video emotion analyzer supporting multiple models"""
    
    def __init__(self, model_type, **kwargs):
        """
        Initialize unified video analyzer
        
        Args:
            model_type: 'deepface', 'ghostfacenetsv2', or 'fer_mtcnn'
            **kwargs: Model-specific parameters
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_type]
        if not config['available']:
            raise ImportError(f"{config['name']} is not available. Install with: {config['install_cmd']}")
        
        self.model_type = model_type
        self.config = config
        self.emotion_classes = config['emotions']
        self.emotion_colors = config['colors']
        
        # Initialize model-specific components
        if model_type == 'deepface':
            self._init_deepface(kwargs.get('model_name', 'fer2013'), 
                               kwargs.get('detector_backend', 'opencv'))
        elif model_type == 'ghostfacenetsv2':
            self._init_ghostfacenetsv2(kwargs.get('checkpoint_path'), 
                                      kwargs.get('image_size', 224),
                                      kwargs.get('device', 'cpu'))
        elif model_type == 'fer_mtcnn':
            self._init_fer_mtcnn(kwargs.get('model_name', 'fer2013'))
        
        print(f"✅ {config['name']} video analyzer initialized")
    
    def _init_deepface(self, model_name, detector_backend):
        """Initialize DeepFace model"""
        self.model_name = model_name
        self.detector_backend = detector_backend
        print(f"  Model: {model_name}")
        print(f"  Detector: {detector_backend}")
    
    def _init_ghostfacenetsv2(self, checkpoint_path, image_size, device):
        """Initialize GhostFaceNetsV2 model"""
        self.device = device
        self.image_size = image_size
        
        # Create GhostFaceNetsV2 model
        self.model = GhostFaceNetsV2(
            image_size=image_size,
            num_classes=len(self.emotion_classes),
            width=1.0,
            dropout=0.2
        )
        
        # Load pretrained weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading GhostFaceNetsV2 weights from {checkpoint_path}")
            try:
                load_pretrained(self.model, checkpoint_path, not_vit=True)
                print("✅ Weights loaded successfully")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
                print("Trying alternative loading method...")
                try:
                    load_pretrained(self.model, checkpoint_path, not_vit=False)
                    print("✅ Weights loaded with alternative method")
                except Exception as e2:
                    print(f"❌ Alternative loading also failed: {e2}")
                    raise Exception(f"Could not load checkpoint: {e}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        # Move model to device and set evaluation mode
        self.model.to(device)
        self.model.eval()
        
        # Initialize face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        print(f"  Device: {device}")
        print(f"  Image size: {image_size}")
    
    def _init_fer_mtcnn(self, model_name):
        """Initialize FER MTCNN model"""
        self.model_name = model_name
        self.detector = FER(mtcnn=True)
        print(f"  Model: {model_name}")
        print(f"  Detector: MTCNN")
    
    def predict_emotion_from_array(self, image_array):
        """Predict emotion from numpy array using the selected model"""
        if self.model_type == 'deepface':
            return self._predict_deepface(image_array)
        elif self.model_type == 'ghostfacenetsv2':
            return self._predict_ghostfacenetsv2(image_array)
        elif self.model_type == 'fer_mtcnn':
            return self._predict_fer_mtcnn(image_array)
    
    def _predict_deepface(self, image_array):
        """DeepFace emotion prediction"""
        try:
            # Save temporary image
            temp_path = "temp_deepface_image.jpg"
            cv2.imwrite(temp_path, image_array)
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Extract emotion predictions
            if isinstance(result, list):
                # Multiple faces detected
                emotions = []
                for face_result in result:
                    emotion_data = face_result['emotion']
                    dominant_emotion = face_result['dominant_emotion']
                    emotions.append({
                        'emotions': emotion_data,
                        'dominant_emotion': dominant_emotion,
                        'confidence': emotion_data[dominant_emotion]
                    })
                return emotions
            else:
                # Single face detected
                emotion_data = result['emotion']
                dominant_emotion = result['dominant_emotion']
                return [{
                    'emotions': emotion_data,
                    'dominant_emotion': dominant_emotion,
                    'confidence': emotion_data[dominant_emotion]
                }]
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return []
    
    def _predict_ghostfacenetsv2(self, image_array):
        """GhostFaceNetsV2 emotion prediction"""
        try:
            # Detect faces using MediaPipe
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            face_emotions = []
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image_array.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    
                    # Extract face region
                    face_roi = image_array[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Preprocess face image
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
                        face_tensor = torch.from_numpy(face_resized).float() / 255.0
                        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = self.model(face_tensor)
                            probabilities = F.softmax(outputs, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            
                            emotion_label = self.emotion_classes[predicted_idx.item()]
                            confidence_score = confidence.item()
                            
                            face_emotions.append({
                                'bbox': (x, y, w, h),
                                'dominant_emotion': emotion_label,
                                'confidence': confidence_score
                            })
            
            return face_emotions
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return []
    
    def _predict_fer_mtcnn(self, image_array):
        """FER MTCNN emotion prediction"""
        try:
            # Detect emotions using FER
            emotions = self.detector.detect_emotions(image_array)
            
            if not emotions:
                return []
            
            # Convert FER format to our format
            results = []
            for emotion_data in emotions:
                # Get bounding box
                bbox = emotion_data['box']
                x, y, w, h = bbox
                
                # Get emotions dictionary
                emotion_scores = emotion_data['emotions']
                
                # Find dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                results.append({
                    'bbox': (x, y, w, h),
                    'emotions': emotion_scores,
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence
                })
            
            return results
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return []
    
    def analyze_video(self, video_path, interval_seconds=30, sample_rate=1):
        """
        Analyze emotions in video
        
        Args:
            video_path: Path to video file
            interval_seconds: Time interval for chart (default: 30 seconds)
            sample_rate: Process every Nth frame (default: 1 = every frame)
        
        Returns:
            Dictionary with time series data
        """
        print(f"Analyzing video with {self.config['name']}: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds ({timedelta(seconds=int(duration))})")
        print(f"  Sample rate: Every {sample_rate} frame(s)")
        
        # Initialize data structures
        emotion_data = defaultdict(list)  # emotion -> list of (timestamp, confidence)
        frame_count = 0
        processed_frames = 0
        
        # Process frames
        print("\nProcessing video frames...")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame based on sample rate
            if frame_count % sample_rate == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                
                try:
                    # Predict emotions
                    emotions = self.predict_emotion_from_array(frame)
                    
                    # Record emotions (use the most confident prediction if multiple faces)
                    if emotions:
                        # Sort by confidence and take the highest
                        emotions.sort(key=lambda x: x['confidence'], reverse=True)
                        best_emotion_data = emotions[0]
                        
                        dominant_emotion = best_emotion_data['dominant_emotion']
                        confidence = best_emotion_data['confidence']
                        
                        emotion_data[dominant_emotion].append((timestamp, confidence))
                    else:
                        # No face detected - record as first emotion class with low confidence
                        emotion_data[self.emotion_classes[0]].append((timestamp, 0.1))
                    
                    processed_frames += 1
                    
                    # Progress update
                    if processed_frames % 10 == 0:
                        elapsed = time.time() - start_time
                        progress = (frame_count / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% ({processed_frames} frames processed in {elapsed:.1f}s)")
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
        
        cap.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Video analysis completed in {total_time:.1f} seconds")
        print(f"  Processed {processed_frames} frames")
        print(f"  Average processing speed: {processed_frames/total_time:.1f} FPS")
        
        return {
            'emotion_data': dict(emotion_data),
            'video_duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames
        }

def get_available_models():
    """Get list of available models"""
    available_models = []
    for model_key, config in MODEL_CONFIGS.items():
        if config['available']:
            available_models.append(model_key)
    return available_models

def analyze_video_file(video_path, model_type, interval_seconds=30, sample_rate=1, output_path=None, **model_kwargs):
    """Complete unified video analysis workflow"""
    print(f"Unified Video Emotion Analysis - {MODEL_CONFIGS[model_type]['name']}")
    print("=" * 60)
    
    # Check if model is available
    if not MODEL_CONFIGS[model_type]['available']:
        print(f"❌ {MODEL_CONFIGS[model_type]['name']} is not available")
        print(f"Install with: {MODEL_CONFIGS[model_type]['install_cmd']}")
        return
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get checkpoint for GhostFaceNetsV2 if needed
    if model_type == 'ghostfacenetsv2' and 'checkpoint_path' not in model_kwargs:
        checkpoints = get_available_checkpoints()
        if checkpoints:
            model_kwargs['checkpoint_path'] = checkpoints[0]
            print(f"Using checkpoint: {os.path.basename(checkpoints[0])}")
        else:
            print("❌ No GhostFaceNetsV2 checkpoints found")
            return
    
    # Initialize analyzer
    try:
        analyzer = UnifiedVideoAnalyzer(model_type, **model_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Analyze video
    analysis_result = analyzer.analyze_video(
        video_path=video_path,
        interval_seconds=interval_seconds,
        sample_rate=sample_rate
    )
    
    if analysis_result is None:
        return
    
    # Generate chart
    if output_path is None:
        output_path = f"{model_type}_emotion_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.png"
    
    print(f"\nGenerating emotion chart...")
    create_emotion_chart(analysis_result, interval_seconds, output_path, 
                        emotion_classes=analyzer.emotion_classes, 
                        emotion_colors=analyzer.emotion_colors, 
                        title_prefix=analyzer.config['name'])
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"  Model: {analyzer.config['name']}")
    print(f"  Video duration: {analysis_result['video_duration']:.1f} seconds")
    print(f"  Total detections: {sum(len(timestamps) for timestamps in analysis_result['emotion_data'].values())}")
    print(f"  Chart saved to: {output_path}")
    
    # Print emotion breakdown
    emotion_counts = {emotion: len(timestamps) for emotion, timestamps in analysis_result['emotion_data'].items()}
    total_detections = sum(emotion_counts.values())
    
    if total_detections > 0:
        print(f"\nEmotion breakdown:")
        for emotion in analyzer.emotion_classes:
            if emotion in emotion_counts:
                percentage = (emotion_counts[emotion] / total_detections) * 100
                print(f"  {emotion}: {emotion_counts[emotion]} ({percentage:.1f}%)")

if __name__ == "__main__":
    print("Unified Video Emotion Analysis")
    print("=" * 50)
    
    # Show available models
    available_models = get_available_models()
    if not available_models:
        print("❌ No emotion recognition models available")
        print("\nInstall required packages:")
        for model_key, config in MODEL_CONFIGS.items():
            print(f"  {config['name']}: {config['install_cmd']}")
        exit(1)
    
    print("✓ Available models:")
    for i, model_key in enumerate(available_models, 1):
        config = MODEL_CONFIGS[model_key]
        print(f"  {i}. {config['name']} ({model_key})")
    
    # Get model choice
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or enter model name: ").strip()
            
            # Try to parse as number
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    model_type = available_models[choice_idx]
                    break
                else:
                    print(f"Invalid choice. Please enter 1-{len(available_models)}")
            else:
                # Try to parse as model name
                if choice in available_models:
                    model_type = choice
                    break
                else:
                    print(f"Unknown model: {choice}")
        except ValueError:
            print("Invalid input. Please try again.")
    
    config = MODEL_CONFIGS[model_type]
    print(f"\nSelected: {config['name']}")
    
    # Get video path
    video_path = input("\nEnter video file path: ").strip()
    
    if not video_path:
        print("No video path provided. Exiting.")
        exit(1)
    
    # Get analysis parameters
    try:
        interval_seconds = int(input("Enter time interval in seconds (default: 30): ") or "30")
        sample_rate = int(input("Enter sample rate (process every Nth frame, default: 1): ") or "1")
    except ValueError:
        print("Invalid input. Using defaults.")
        interval_seconds = 30
        sample_rate = 1
    
    # Get model-specific parameters
    model_kwargs = {}
    if model_type == 'deepface':
        detector_backend = input("Enter detector backend (opencv/mtcnn/retinaface, default: opencv): ").strip() or "opencv"
        model_kwargs['detector_backend'] = detector_backend
    elif model_type == 'ghostfacenetsv2':
        device = input("Enter device (cpu/cuda, default: auto): ").strip()
        if device:
            model_kwargs['device'] = device
        else:
            model_kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get output path
    output_path = input("Enter output chart path (press Enter for auto): ").strip()
    if not output_path:
        output_path = None
    
    # Run analysis
    analyze_video_file(
        video_path=video_path,
        model_type=model_type,
        interval_seconds=interval_seconds,
        sample_rate=sample_rate,
        output_path=output_path,
        **model_kwargs
    ) 