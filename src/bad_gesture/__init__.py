#!/usr/bin/env python3
"""
Bad Gesture Detection Module
Supports both 2D and 3D detection methods
"""

from .bad_gestures_2d import BadGestureDetector
from .bad_gestures_3d import BadGestureDetector3D

__all__ = ['BadGestureDetector', 'BadGestureDetector3D']
