"""
Multicam Screen Change Detector - CANNY EDGE VERSION

Uses Canny edge detection to find grid dividers.
This is much more robust for detecting subtle divider lines.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Change detection
STABILITY_THRESHOLD = 2  # Only need 2 consecutive frames to confirm a change

# Quadrant activity detection
BRIGHTNESS_THRESHOLD = 30  # Threshold for black screen detection
EDGE_THRESHOLD = 0.1  # Threshold for edge density in quadrants

# Canny edge detection parameters
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Grid layout detection
VERTICAL_DIVIDER_MIN_LENGTH = 0.4  # 40% of height
HORIZONTAL_DIVIDER_MIN_LENGTH = 0.4  # 40% of width
DIVIDER_THICKNESS_TOLERANCE = 5  # pixels around center line to check
EDGE_DENSITY_THRESHOLD = 0.3  # 30% of sampled points must have edges


@dataclass
class StateFingerprint:
    """Represents the state of the video screen at a given frame."""
    layout: str  # 'grid' or 'fullscreen'
    quadrants: List[int]  # 1 for active, 0 for inactive
    fingerprint: str  # Unique identifier for this state

    def to_dict(self):
        return asdict(self)


@dataclass
class StateChange:
    """Represents a detected change in screen state."""
    time: float  # Time in seconds
    frame_number: int
    from_state: StateFingerprint
    to_state: StateFingerprint

    def to_dict(self):
        return {
            'time': self.time,
            'frame_number': self.frame_number,
            'from_state': self.from_state.to_dict(),
            'to_state': self.to_state.to_dict()
        }


class ScreenStateDetector:
    """
    Detects changes in multicam screen layout using Canny edge detection.

    Key insight: Grid dividers create strong continuous vertical and horizontal
    edges that Canny can detect reliably.
    """

    def __init__(self, video_path: str, duration_override: Optional[float] = None, debug: bool = False):
        """Initialize the detector with a video file."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.debug = debug

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if duration_override and duration_override > 0:
            self.duration = duration_override
            self.fps = self.total_frames / self.duration if self.duration > 0 else 30
            self.fps_source = "calculated_from_duration"
            print(f"Using provided duration: {duration_override}s")
            print(f"Calculated FPS: {self.fps:.2f} (from {self.total_frames} frames)")
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.fps_source = "opencv_detected"

        # State tracking
        self.changes: List[StateChange] = []
        self.current_state: Optional[StateFingerprint] = None
        self.previous_state: Optional[StateFingerprint] = None
        self.stable_state_fingerprint: Optional[str] = None
        self.stable_frames: int = 0
        self.frame_count: int = 0

    def detect_layout_canny(self, frame: np.ndarray) -> Tuple[str, dict]:
        """
        Detect layout using Canny edge detection.
        
        Strategy:
        1. Apply Canny edge detection to find all edges
        2. Check for strong vertical edge line at center
        3. Check for strong horizontal edge line at center
        4. If both present with sufficient length → grid
        5. Otherwise → fullscreen
        
        Returns:
            Tuple of (layout_type, debug_info)
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        
        center_x = width // 2
        center_y = height // 2
        
        debug_info = {}
        
        # Check vertical divider (with tolerance for thickness)
        vertical_edge_count = 0
        vertical_samples = 0
        
        y_start = int(height * 0.1)
        y_end = int(height * 0.9)
        
        for y in range(y_start, y_end, 2):  # Sample every 2 pixels
            # Check a band around the center (not just a single line)
            has_edge = False
            for x_offset in range(-DIVIDER_THICKNESS_TOLERANCE,
                                 DIVIDER_THICKNESS_TOLERANCE + 1):
                x = center_x + x_offset
                if 0 <= x < width:
                    if edges[y, x] > 0:
                        has_edge = True
                        break
            
            if has_edge:
                vertical_edge_count += 1
            vertical_samples += 1
        
        vertical_edge_density = vertical_edge_count / vertical_samples if vertical_samples > 0 else 0
        
        # Check horizontal divider (with tolerance for thickness)
        horizontal_edge_count = 0
        horizontal_samples = 0
        
        x_start = int(width * 0.1)
        x_end = int(width * 0.9)
        
        for x in range(x_start, x_end, 2):  # Sample every 2 pixels
            # Check a band around the center
            has_edge = False
            for y_offset in range(-DIVIDER_THICKNESS_TOLERANCE,
                                 DIVIDER_THICKNESS_TOLERANCE + 1):
                y = center_y + y_offset
                if 0 <= y < height:
                    if edges[y, x] > 0:
                        has_edge = True
                        break
            
            if has_edge:
                horizontal_edge_count += 1
            horizontal_samples += 1
        
        horizontal_edge_density = horizontal_edge_count / horizontal_samples if horizontal_samples > 0 else 0
        
        debug_info['vertical_edge_density'] = vertical_edge_density
        debug_info['horizontal_edge_density'] = horizontal_edge_density
        debug_info['vertical_edge_count'] = vertical_edge_count
        debug_info['horizontal_edge_count'] = horizontal_edge_count
        
        # Decision: Grid if both dividers detected
        has_vertical_divider = vertical_edge_density >= EDGE_DENSITY_THRESHOLD
        has_horizontal_divider = horizontal_edge_density >= EDGE_DENSITY_THRESHOLD
        
        is_grid = has_vertical_divider and has_horizontal_divider
        
        debug_info['has_vertical_divider'] = has_vertical_divider
        debug_info['has_horizontal_divider'] = has_horizontal_divider
        debug_info['is_grid'] = is_grid
        
        return ('grid' if is_grid else 'fullscreen'), debug_info

    def detect_layout(self, frame: np.ndarray) -> str:
        """Wrapper for Canny-based layout detection."""
        layout, debug_info = self.detect_layout_canny(frame)
        
        if self.debug and self.frame_count % 100 == 0:
            print(f"\n[Frame {self.frame_count}] Layout: {layout}")
            print(f"  V-edge density: {debug_info['vertical_edge_density']:.3f} (count: {debug_info['vertical_edge_count']})")
            print(f"  H-edge density: {debug_info['horizontal_edge_density']:.3f} (count: {debug_info['horizontal_edge_count']})")
            print(f"  Has V-divider: {debug_info['has_vertical_divider']}")
            print(f"  Has H-divider: {debug_info['has_horizontal_divider']}")
        
        return layout

    def is_quadrant_active(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """
        Check if a quadrant has active video content using histogram analysis.

        Active video content typically has:
        - Wider distribution of brightness values (high std deviation)
        - Multiple peaks in histogram (varied content)
        - Less concentration in black/dark ranges

        Inactive/black screens have:
        - Most pixels clustered in low values (0-30)
        - Very low standard deviation
        - Single dominant peak near zero
        """
        # Extract the quadrant region (keep color for better analysis)
        height, width = frame.shape[:2]

        # Ensure we stay within bounds
        x_end = min(x + w, width)
        y_end = min(y + h, height)

        if x >= x_end or y >= y_end:
            return False

        # Extract quadrant region in color and grayscale
        region_color = frame[y:y_end, x:x_end]
        region_gray = cv2.cvtColor(region_color, cv2.COLOR_BGR2GRAY)

        if region_gray.size == 0:
            return False

        # Strategy 1: Histogram Analysis
        # Calculate histogram (256 bins for grayscale values 0-255)
        hist = cv2.calcHist([region_gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / region_gray.size  # Normalize to get probability

        # Calculate histogram metrics
        # 1. How much content is in the "black" range (0-30)
        black_ratio = np.sum(hist[:30])

        # 2. Standard deviation of pixel values (spread of brightness)
        std_dev = np.std(region_gray)

        # 3. Histogram entropy (measure of randomness/variety)
        # Remove zeros to avoid log(0)
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero)) if len(hist_nonzero) > 0 else 0

        # 4. Color variety check (for color frame)
        # Calculate standard deviation across color channels
        color_std = np.mean([np.std(region_color[:,:,i]) for i in range(3)])

        # 5. Number of significant peaks in histogram
        # Smooth histogram and find peaks
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist, sigma=2)
        # Count bins with >1% of pixels
        significant_bins = np.sum(hist_smooth > 0.01)

        # Decision logic with multiple criteria
        is_active = False
        reasons = []

        # Check 1: Not mostly black
        if black_ratio < 0.8:  # Less than 80% black pixels
            reasons.append("not_black")
            is_active = True

        # Check 2: Has brightness variation
        if std_dev > 15:  # Significant brightness variation
            reasons.append("high_std")
            is_active = True

        # Check 3: Has color variation
        if color_std > 10:  # Significant color variation
            reasons.append("color_variety")
            is_active = True

        # Check 4: Histogram entropy (variety of values)
        if entropy > 4.0:  # High entropy = diverse content
            reasons.append("high_entropy")
            is_active = True

        # Check 5: Multiple brightness levels used
        if significant_bins > 10:  # Uses many different brightness levels
            reasons.append("multiple_peaks")
            is_active = True

        # Fallback: Check average brightness (original method)
        avg_brightness = np.mean(region_gray)
        if not is_active and avg_brightness > BRIGHTNESS_THRESHOLD:
            reasons.append("bright")
            is_active = True

        # Edge detection as final check (original method)
        if not is_active:
            edges = cv2.Canny(region_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > EDGE_THRESHOLD:
                reasons.append("edges")
                is_active = True

        if self.debug and self.frame_count % 500 == 0:
            print(f"    Quadrant ({x},{y},{w}x{h}):")
            print(f"      Black ratio: {black_ratio:.2f}, Std: {std_dev:.1f}, Entropy: {entropy:.2f}")
            print(f"      Color std: {color_std:.1f}, Significant bins: {significant_bins}")
            print(f"      Active: {is_active} ({', '.join(reasons)})")

        return is_active

    def get_state_fingerprint(self, frame: np.ndarray) -> StateFingerprint:
        """Generate a state fingerprint for the current frame."""
        height, width = frame.shape[:2]
        layout = self.detect_layout(frame)

        quadrants = [0, 0, 0, 0]

        if layout == 'grid':
            # Check all 4 quadrants
            half_w = width // 2
            half_h = height // 2

            # Top-left, Top-right, Bottom-left, Bottom-right
            quadrants[0] = 1 if self.is_quadrant_active(frame, 0, 0, half_w, half_h) else 0
            quadrants[1] = 1 if self.is_quadrant_active(frame, half_w, 0, half_w, half_h) else 0
            quadrants[2] = 1 if self.is_quadrant_active(frame, 0, half_h, half_w, half_h) else 0
            quadrants[3] = 1 if self.is_quadrant_active(frame, half_w, half_h, half_w, half_h) else 0
        else:
            # Fullscreen - only check if there's content
            quadrants[0] = 1 if self.is_quadrant_active(frame, 0, 0, width, height) else 0

        fingerprint = f"{layout}-{''.join(map(str, quadrants))}"

        return StateFingerprint(
            layout=layout,
            quadrants=quadrants,
            fingerprint=fingerprint
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[StateChange]:
        """Process a single frame and detect state changes."""
        state = self.get_state_fingerprint(frame)
        self.current_state = state

        change_detected = None

        # Check for changes
        if self.previous_state and state.fingerprint != self.previous_state.fingerprint:
            # State has changed - wait for stability
            if self.stable_state_fingerprint == state.fingerprint:
                self.stable_frames += 1

                # If stable for enough frames, register the change
                if self.stable_frames >= STABILITY_THRESHOLD:
                    change_detected = StateChange(
                        time=timestamp,
                        frame_number=self.frame_count,
                        from_state=self.previous_state,
                        to_state=state
                    )
                    self.changes.append(change_detected)
                    self.previous_state = state
                    self.stable_frames = 0
            else:
                # New potential state
                self.stable_state_fingerprint = state.fingerprint
                self.stable_frames = 1
        else:
            # State unchanged
            self.stable_frames = 0
            self.stable_state_fingerprint = None

        if not self.previous_state:
            self.previous_state = state

        self.frame_count += 1
        return change_detected

    def process_video(self, show_preview: bool = False, save_output: bool = True) -> List[StateChange]:
        """Process the entire video and detect all state changes."""
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f} ({self.fps_source})")
        print(f"Duration: {self.duration:.2f}s ({self._format_time(self.duration)})")
        print(f"Stability threshold: {STABILITY_THRESHOLD} frames (immediate detection)")
        print(f"Detection method: Canny edge detection")
        print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = self.frame_count / self.fps
            change = self.process_frame(frame, timestamp)

            if change:
                print(f"Change detected at {self._format_time(timestamp)}")
                print(f"  From: {change.from_state.fingerprint}")
                print(f"  To:   {change.to_state.fingerprint}")

            # Show preview if requested
            if show_preview:
                self._draw_state_overlay(frame)
                cv2.imshow('Screen State Detector (Canny)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress update
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames} frames)", end='\r')

        print("\n" + "-" * 60)
        print(f"Processing complete! Detected {len(self.changes)} changes.")

        if save_output:
            self._save_results()

        self.cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        return self.changes

    def _draw_state_overlay(self, frame: np.ndarray):
        """Draw current state information on the frame."""
        if not self.current_state:
            return

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_offset = 40
        cv2.putText(frame, f"Layout: {self.current_state.layout}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 30
        if self.current_state.layout == 'grid':
            active_count = sum(self.current_state.quadrants)
            cv2.putText(frame, f"Active Cameras: {active_count}/4",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw quadrant grid
            y_offset += 30
            for i, active in enumerate(self.current_state.quadrants):
                color = (0, 255, 0) if active else (0, 0, 255)
                text = f"Q{i+1}: {'ON' if active else 'OFF'}"
                x = 20 + (i % 2) * 100
                y = y_offset + (i // 2) * 25
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            status = 'Active' if self.current_state.quadrants[0] else 'Inactive'
            cv2.putText(frame, f"Status: {status}",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS.MS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 100)
        return f"{mins}:{secs:02d}.{ms:02d}"

    def _save_results(self):
        """Save detection results to JSON file."""
        video_name = Path(self.video_path).stem
        output_path = Path(self.video_path).parent / f"{video_name}_screen_changes_canny.json"

        results = {
            'video_path': self.video_path,
            'duration': self.duration,
            'fps': self.fps,
            'fps_source': self.fps_source,
            'total_frames': self.total_frames,
            'detection_method': 'canny_edge_detection',
            'changes_detected': len(self.changes),
            'detection_params': {
                'stability_threshold': STABILITY_THRESHOLD,
                'canny_low': CANNY_LOW_THRESHOLD,
                'canny_high': CANNY_HIGH_THRESHOLD,
                'edge_density_threshold': EDGE_DENSITY_THRESHOLD,
            },
            'changes': [change.to_dict() for change in self.changes]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    """Command-line interface for the screen state detector."""
    parser = argparse.ArgumentParser(
        description='Detect layout and state changes using Canny edge detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Duration format examples:
  --duration 1792           # 1792 seconds
  --duration 29:52          # 29 minutes 52 seconds
  --duration 1:29:52        # 1 hour 29 minutes 52 seconds
        """
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview with state overlay')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to JSON file')
    parser.add_argument('--duration', type=str, default=None,
                       help='Actual video duration in seconds or MM:SS or HH:MM:SS format')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed debug output')

    args = parser.parse_args()

    # Parse duration if provided
    duration_override = None
    if args.duration:
        try:
            if ':' in args.duration:
                parts = args.duration.split(':')
                if len(parts) == 2:
                    duration_override = int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    duration_override = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                else:
                    print(f"Invalid duration format: {args.duration}")
                    return 1
            else:
                duration_override = float(args.duration)
        except ValueError:
            print(f"Invalid duration value: {args.duration}")
            return 1

    detector = ScreenStateDetector(args.video_path, duration_override=duration_override, debug=args.debug)
    changes = detector.process_video(
        show_preview=args.preview,
        save_output=not args.no_save
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, change in enumerate(changes, 1):
        print(f"\nChange #{i} at {detector._format_time(change.time)}")
        print(f"  From: {change.from_state.layout} - {change.from_state.quadrants}")
        print(f"  To:   {change.to_state.layout} - {change.to_state.quadrants}")


if __name__ == '__main__':
    main()