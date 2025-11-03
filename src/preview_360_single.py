#!/usr/bin/env python3
"""
Quick 360° Video Preview for Single File
Press arrow keys to adjust angle, SPACE to print FFmpeg command, ESC to exit
"""

import cv2
import numpy as np
import sys

def equirect_to_perspective(equirect_img, yaw, pitch, fov):
    """Convert equirectangular to perspective view"""
    h, w = equirect_img.shape[:2]
    out_h, out_w = 720, 1280

    # Convert angles to radians
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    fov_rad = np.deg2rad(fov)

    # Create output coordinate grid
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]

    # Normalize to [-1, 1]
    x_norm = (x_coords - out_w / 2) / (out_w / 2)
    y_norm = (y_coords - out_h / 2) / (out_h / 2)

    # Apply FOV
    x_norm *= np.tan(fov_rad / 2)
    y_norm *= np.tan(fov_rad / 2) * (out_h / out_w)

    # Convert to 3D rays
    z = np.ones_like(x_norm)
    x = x_norm
    y = y_norm

    # Normalize rays
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    # Apply pitch rotation
    y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
    z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)

    # Apply yaw rotation
    x_final = x * np.cos(yaw_rad) + z_rot * np.sin(yaw_rad)
    z_final = -x * np.sin(yaw_rad) + z_rot * np.cos(yaw_rad)

    # Convert to equirectangular coordinates
    theta = np.arctan2(x_final, z_final)
    phi = np.arcsin(np.clip(y_rot, -1, 1))

    # Map to pixel coordinates
    equirect_x = ((theta / np.pi + 1) * w / 2).astype(np.float32)
    equirect_y = ((phi / (np.pi / 2) + 1) * h / 2).astype(np.float32)

    # Remap
    result = cv2.remap(equirect_img, equirect_x, equirect_y,
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python preview_360_single.py <video_file>")
        print("Example: python preview_360_single.py VID_20250709_203100_00_010.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"\n📹 Opening 360° video: {video_path}")
    print("\n🎮 Controls:")
    print("  ← → (or A D): Adjust Yaw (horizontal rotation)")
    print("  ↑ ↓ (or W S): Adjust Pitch (vertical tilt)")
    print("  + -  : Adjust FOV (zoom)")
    print("  SPACE: Print FFmpeg command with current settings")
    print("  ESC  : Exit\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        sys.exit(1)

    # Get middle frame and keep it
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, original_frame = cap.read()

    if not ret:
        print(f"❌ Error: Cannot read frame from video")
        cap.release()
        sys.exit(1)

    # Initial settings
    yaw = 0
    pitch = 0
    fov = 120

    print(f"✅ Loaded frame {total_frames // 2} / {total_frames}")
    print("🎮 Window is ready! Make sure the preview window is in focus (click on it)")

    while True:
        # Use the same frame, just re-project it with new angles
        # Convert to perspective
        preview_frame = equirect_to_perspective(original_frame.copy(), yaw, pitch, fov)

        # Add overlay
        cv2.putText(preview_frame, f"Yaw: {yaw:3d}  Pitch: {pitch:3d}  FOV: {fov:3d}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(preview_frame, "SPACE=Show Command | ESC=Exit",
                   (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("360° Preview - Adjust with Arrow Keys", preview_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("👋 Exiting...")
            break
        elif key == ord(' '):  # SPACE
            print("\n" + "="*70)
            print("📋 FFmpeg Command to Convert with Current Settings:")
            print("="*70)
            output_file = video_path.replace('.mp4', f'_converted_yaw{yaw}_pitch{pitch}.mp4')
            cmd = f'''ffmpeg -i "{video_path}" \\
  -vf "v360=e:flat:iv_fov=360:ih_fov=180:yaw={yaw}:pitch={pitch}:roll=0:w=1920:h=1080:interp=linear" \\
  -c:v libx264 -preset medium -crf 23 -c:a copy \\
  "{output_file}"'''
            print(cmd)
            print("\n📁 Output file will be:", output_file)
            print("="*70 + "\n")
        elif key == 81 or key == ord('a') or key == ord('A'):  # Left arrow or A
            yaw = (yaw - 5) % 360
            print(f"← Yaw: {yaw}°")
        elif key == 83 or key == ord('d') or key == ord('D'):  # Right arrow or D
            yaw = (yaw + 5) % 360
            print(f"→ Yaw: {yaw}°")
        elif key == 82 or key == ord('w') or key == ord('W'):  # Up arrow or W
            pitch = max(-90, pitch + 5)
            print(f"↑ Pitch: {pitch}°")
        elif key == 84 or key == ord('s') or key == ord('S'):  # Down arrow or S
            pitch = min(90, pitch - 5)
            print(f"↓ Pitch: {pitch}°")
        elif key == ord('+') or key == ord('='):
            fov = min(180, fov + 5)
            print(f"+ FOV: {fov}°")
        elif key == ord('-') or key == ord('_'):
            fov = max(30, fov - 5)
            print(f"- FOV: {fov}°")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
