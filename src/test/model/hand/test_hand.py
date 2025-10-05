import unittest
import cv2
import numpy as np
import sys
import os
import json

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hand.mediapipe import MediaPipeHandDetector
from src.models.hand.yolo import YOLOHandDetector


class TestHandDetection(unittest.TestCase):
    """Test cases for hand detection models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.test_image_path = "test/images/test_hand.jpg"
        cls.ground_truth_path = "test/images/test_hand_ground_truth.json"
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(cls.test_image_path):
            cls._create_test_image()
        
        # Load test image
        cls.test_image = cv2.imread(cls.test_image_path)
        if cls.test_image is None:
            raise FileNotFoundError(f"Could not load test image: {cls.test_image_path}")
    
    @classmethod
    def _create_test_image(cls):
        """Create a simple test image for hand detection"""
        # Create a simple colored image (this would be replaced with actual hand image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some simple shapes to simulate a hand
        cv2.rectangle(test_image, (200, 150), (400, 350), (100, 150, 200), -1)
        cv2.circle(test_image, (300, 200), 30, (150, 200, 250), -1)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        cv2.imwrite(cls.test_image_path, test_image)
        print(f"Created test image: {cls.test_image_path}")
    
    def test_mediapipe_hand_detection(self):
        """Test MediaPipe hand detection"""
        print("\n=== Testing MediaPipe Hand Detection ===")
        
        detector = MediaPipeHandDetector()
        
        # Test detection
        results = detector.detect(self.test_image)
        self.assertIsNotNone(detector.model, "MediaPipe hand model should be initialized")
        
        # Test conversion to dict
        left_hand, right_hand = detector.convert_to_dict(results)
        print(f"Left hand detected: {left_hand is not None}")
        print(f"Right hand detected: {right_hand is not None}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_landmarks(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "hand")
        self.assertEqual(model_info["name"], "MediaPipe Hands")
        
        # Save results as ground truth
        ground_truth = {
            "model": "mediapipe",
            "left_hand": left_hand,
            "right_hand": right_hand,
            "model_info": model_info
        }
        
        self._save_ground_truth("mediapipe", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def test_yolo_hand_detection(self):
        """Test YOLO hand detection"""
        print("\n=== Testing YOLO Hand Detection ===")
        
        detector = YOLOHandDetector()
        
        # Test detection (might fail if model doesn't exist, which is expected)
        results = detector.detect(self.test_image)
        
        if detector.model is None:
            print("YOLO hand model not available - this is expected if model file doesn't exist")
            self.skipTest("YOLO hand model not available")
        
        # Test conversion to dict
        left_hand, right_hand = detector.convert_to_dict(results)
        print(f"Left hand detected: {left_hand is not None}")
        print(f"Right hand detected: {right_hand is not None}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_landmarks(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "hand")
        self.assertEqual(model_info["name"], "YOLO Hand Detector")
        
        # Save results as ground truth
        ground_truth = {
            "model": "yolo",
            "left_hand": left_hand,
            "right_hand": right_hand,
            "model_info": model_info
        }
        
        self._save_ground_truth("yolo", ground_truth)
        
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
            
            # Test MediaPipe consistency
            if "mediapipe" in ground_truth:
                detector = MediaPipeHandDetector()
                results = detector.detect(self.test_image)
                left_hand, right_hand = detector.convert_to_dict(results)
                
                gt_left = ground_truth["mediapipe"]["left_hand"]
                gt_right = ground_truth["mediapipe"]["right_hand"]
                
                # Compare presence of hands (basic consistency check)
                self.assertEqual((left_hand is not None), (gt_left is not None), 
                               "Left hand detection should be consistent")
                self.assertEqual((right_hand is not None), (gt_right is not None), 
                               "Right hand detection should be consistent")
                
                detector.cleanup()
                print("MediaPipe consistency check passed")
            
        except Exception as e:
            self.fail(f"Ground truth comparison failed: {e}")


def generate_ground_truth():
    """Generate ground truth data by running models on test image"""
    print("=== Generating Ground Truth for Hand Detection ===")
    
    # Create test suite and run tests to generate ground truth
    suite = unittest.TestSuite()
    suite.addTest(TestHandDetection('test_mediapipe_hand_detection'))
    suite.addTest(TestHandDetection('test_yolo_hand_detection'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hand detection models')
    parser.add_argument('--generate-ground-truth', action='store_true',
                       help='Generate ground truth data')
    
    args = parser.parse_args()
    
    if args.generate_ground_truth:
        generate_ground_truth()
    else:
        unittest.main(verbosity=2)