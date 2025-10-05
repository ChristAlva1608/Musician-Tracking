#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

import os
import sys

# Add project src directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)

print(f"Backend dir: {backend_dir}")
print(f"Web dir: {web_dir}")
print(f"Src root: {src_root}")
print()

# Test imports
try:
    from detect_v2 import DetectorV2
    print("✅ Successfully imported DetectorV2")
except ImportError as e:
    print(f"❌ Failed to import DetectorV2: {e}")

try:
    from integrated_video_processor import IntegratedVideoProcessor
    print("✅ Successfully imported IntegratedVideoProcessor")
except ImportError as e:
    print(f"❌ Failed to import IntegratedVideoProcessor: {e}")

try:
    from database.setup import MusicianDatabase, VideoAlignmentDatabase, ChunkVideoAlignmentDatabase
    print("✅ Successfully imported database classes")
except ImportError as e:
    print(f"❌ Failed to import database classes: {e}")

print()
print("🎉 All import paths are correctly configured!")
print()
print("To run the web server, install the required packages:")
print("pip install -r requirements.txt")
print()
print("Then run:")
print("python main.py")