#!/usr/bin/env python3
"""
Generate ground truth data for all model tests.

This script runs all model tests to generate baseline ground truth data
that will be used for regression testing.
"""

import sys
import os
import subprocess

def run_test_and_generate_groundtruth(test_file, test_type):
    """Run a test file and generate ground truth"""
    print(f"\n{'='*60}")
    print(f"Generating ground truth for {test_type}")
    print(f"{'='*60}")
    
    try:
        # Run the test with ground truth generation flag
        result = subprocess.run([
            sys.executable, test_file, '--generate-ground-truth'
        ], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"⚠️  {test_type} test completed with warnings (return code: {result.returncode})")
        else:
            print(f"✅ {test_type} ground truth generated successfully")
            
    except Exception as e:
        print(f"❌ Error running {test_type} test: {e}")

def main():
    """Main function to generate all ground truth data"""
    print("🔧 Generating Ground Truth Data for Model Tests")
    print("This script will run all model tests to create baseline data.")
    
    # Test files and their descriptions
    tests = [
        ("test/test_hand.py", "Hand Detection"),
        ("test/test_pose.py", "Pose Detection"),
        ("test/test_face.py", "Face Detection"),
        ("test/test_emotion.py", "Emotion Detection")
    ]
    
    # Ensure we're in the right directory
    if not os.path.exists("test"):
        print("❌ Test directory not found. Please run from project root.")
        sys.exit(1)
    
    # Run each test
    for test_file, test_type in tests:
        if os.path.exists(test_file):
            run_test_and_generate_groundtruth(test_file, test_type)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    print(f"\n{'='*60}")
    print("🎉 Ground Truth Generation Complete!")
    print("You can now run normal tests with:")
    print("  python -m pytest test/ -v")
    print("or individual tests with:")
    print("  python test/test_hand.py")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()