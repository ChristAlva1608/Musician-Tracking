import unittest
import cv2
import numpy as np
import sys
import os
import json

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.emotion.deepface import DeepFaceEmotionDetector
from src.models.emotion.ghostfacenet import GhostFaceNetEmotionDetector


class TestEmotionDetection(unittest.TestCase):
    """Test cases for emotion detection models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.test_image_path = "test/images/test_emotion.jpg"
        cls.ground_truth_path = "test/images/test_emotion_ground_truth.json"
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(cls.test_image_path):
            cls._create_test_image()
        
        # Load test image
        cls.test_image = cv2.imread(cls.test_image_path)
        if cls.test_image is None:
            raise FileNotFoundError(f"Could not load test image: {cls.test_image_path}")
    
    @classmethod
    def _create_test_image(cls):
        """Create a simple test image for emotion detection"""
        # Create a simple colored image (this would be replaced with actual face image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add detailed face-like shape for emotion detection
        # Face outline
        cv2.ellipse(test_image, (320, 240), (100, 120), 0, 0, 360, (220, 200, 180), -1)
        # Eyes
        cv2.ellipse(test_image, (280, 210), (15, 8), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(test_image, (360, 210), (15, 8), 0, 0, 360, (50, 50, 50), -1)
        # Eye pupils
        cv2.circle(test_image, (280, 210), 5, (0, 0, 0), -1)
        cv2.circle(test_image, (360, 210), 5, (0, 0, 0), -1)
        # Nose
        cv2.ellipse(test_image, (320, 240), (8, 15), 0, 0, 360, (180, 160, 140), -1)
        # Mouth (neutral expression)
        cv2.ellipse(test_image, (320, 280), (25, 8), 0, 0, 180, (150, 100, 100), -1)
        # Eyebrows
        cv2.ellipse(test_image, (280, 190), (20, 5), 15, 0, 180, (100, 80, 60), -1)
        cv2.ellipse(test_image, (360, 190), (20, 5), -15, 0, 180, (100, 80, 60), -1)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        cv2.imwrite(cls.test_image_path, test_image)
        print(f"Created test image: {cls.test_image_path}")
    
    def test_deepface_emotion_detection(self):
        """Test DeepFace emotion detection"""
        print("\n=== Testing DeepFace Emotion Detection ===")
        
        detector = DeepFaceEmotionDetector()
        
        # Test detection (might fail if dependencies not installed)
        if detector.detector is None:
            print("DeepFace emotion detector not available - this is expected if dependencies not installed")
            self.skipTest("DeepFace emotion detector not available")
        
        results = detector.detect(self.test_image)
        
        # Test conversion to dict
        emotion_data = detector.convert_to_dict(results)
        print(f"Emotion detection results: {emotion_data}")
        
        if emotion_data:
            print(f"Dominant emotion: {emotion_data.get('dominant_emotion', 'unknown')}")
            print(f"Confidence: {emotion_data.get('confidence', 0.0):.3f}")
            print(f"Face detected: {emotion_data.get('face_detected', False)}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_results(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "emotion")
        self.assertEqual(model_info["name"], "DeepFace Emotion Detector")
        
        # Save results as ground truth
        ground_truth = {
            "model": "deepface",
            "emotion_data": emotion_data,
            "model_info": model_info
        }
        
        self._save_ground_truth("deepface", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def test_ghostfacenet_emotion_detection(self):
        """Test GhostFaceNet emotion detection"""
        print("\n=== Testing GhostFaceNet Emotion Detection ===")
        
        detector = GhostFaceNetEmotionDetector()
        
        # Test detection (might fail if dependencies not installed)
        if detector.detector is None:
            print("GhostFaceNet emotion detector not available - this is expected if dependencies not installed")
            self.skipTest("GhostFaceNet emotion detector not available")
        
        results = detector.detect(self.test_image)
        
        # Test conversion to dict
        emotion_data = detector.convert_to_dict(results)
        print(f"Emotion detection results: {emotion_data}")
        
        if emotion_data:
            print(f"Dominant emotion: {emotion_data.get('dominant_emotion', 'unknown')}")
            print(f"Confidence: {emotion_data.get('confidence', 0.0):.3f}")
            print(f"Face detected: {emotion_data.get('face_detected', False)}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_results(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "emotion")
        self.assertEqual(model_info["name"], "GhostFaceNet Emotion Detector")
        
        # Save results as ground truth
        ground_truth = {
            "model": "ghostfacenet",
            "emotion_data": emotion_data,
            "model_info": model_info
        }
        
        self._save_ground_truth("ghostfacenet", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def _save_ground_truth(self, model_name, ground_truth):
        """Save ground truth results to file"""
        try:
            # Load existing ground truth if it exists
            if os.path.exists(self.ground_truth_path):
                with open(self.ground_truth_path, 'r') as f:
                    all_ground_truth = json.load(f)
            else:
                all_ground_truth = {}
            
            # Update with new results
            all_ground_truth[model_name] = ground_truth
            
            # Save updated ground truth
            with open(self.ground_truth_path, 'w') as f:
                json.dump(all_ground_truth, f, indent=2)
            
            print(f"Ground truth saved for {model_name}")
            
        except Exception as e:
            print(f"Failed to save ground truth: {e}")
    
    def test_compare_with_ground_truth(self):
        """Compare current results with saved ground truth"""
        if not os.path.exists(self.ground_truth_path):
            self.skipTest("No ground truth file available")
        
        print("\n=== Comparing with Ground Truth ===")
        
        try:
            with open(self.ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            
            # Test DeepFace consistency (if available)
            if "deepface" in ground_truth:
                detector = DeepFaceEmotionDetector()
                if detector.detector is not None:
                    results = detector.detect(self.test_image)
                    emotion_data = detector.convert_to_dict(results)
                    
                    gt_emotion_data = ground_truth["deepface"]["emotion_data"]
                    
                    # Compare presence of emotion detection (basic consistency check)
                    self.assertEqual((emotion_data is not None), (gt_emotion_data is not None), 
                                   "Emotion detection should be consistent")
                    
                    if emotion_data and gt_emotion_data:
                        # Check if face detection is consistent
                        self.assertEqual(emotion_data.get('face_detected', False), 
                                       gt_emotion_data.get('face_detected', False),
                                       "Face detection should be consistent")
                    
                    detector.cleanup()
                    print("DeepFace consistency check passed")
                else:
                    print("DeepFace model not available for consistency check")
            
        except Exception as e:
            self.fail(f"Ground truth comparison failed: {e}")


def generate_ground_truth():
    """Generate ground truth data by running models on test image"""
    print("=== Generating Ground Truth for Emotion Detection ===")
    
    # Create test suite and run tests to generate ground truth
    suite = unittest.TestSuite()
    suite.addTest(TestEmotionDetection('test_deepface_emotion_detection'))
    suite.addTest(TestEmotionDetection('test_ghostfacenet_emotion_detection'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test emotion detection models')
    parser.add_argument('--generate-ground-truth', action='store_true',
                       help='Generate ground truth data')
    
    args = parser.parse_args()
    
    if args.generate_ground_truth:
        generate_ground_truth()
    else:
        unittest.main(verbosity=2)