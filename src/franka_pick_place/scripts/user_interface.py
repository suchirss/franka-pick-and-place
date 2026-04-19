#!/usr/bin/env python3

"""
ROS2 User Interface for Franka Pick and Place System

Provides a menu-driven interface for:
- Camera calibration and testing
- Cube detection on ArUco grid
- Franka pick and place operations
"""

import time
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import glob
import sys
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# Add package paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(PKG_DIR, 'config')
CALIBRATION_DIR = os.path.join(CONFIG_DIR, 'calibration_images')
CAMERA_PARAMS_FILE = os.path.join(CONFIG_DIR, 'camera_params.yaml')
sys.path.append(PKG_DIR)

from cv_transform.warp_plane import WarpPlane

# Constants
ARUCO_DICT = cv2.aruco.DICT_4X4_250
GRID_MARKER_ID_ORIGIN = 15
CUBE_MARKER_IDS = list(range(70, 76))
MARKER_SIZE_M = 0.0381  # 1.5 inch in meters
CHECKERBOARD_SIZE = (5, 7)
SQUARE_SIZE = 0.0381

_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_sigint)


def load_camera_params(yaml_file='camera_params.yaml'):
    """Load camera matrix and distortion coefficients from YAML file."""
    pkg_share = get_package_share_directory('franka_pick_place')
    yaml_path = os.path.join(pkg_share, 'config', yaml_file)
    
    try:
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data['camera_matrix']).reshape((3, 3))
        dist_coeffs = np.array(calib_data['distortion_coefficients']).reshape((-1, 1))
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"[ERROR] {yaml_path} not found! Run calibration first.")
        return None, None


def capture_calibration_images(save_dir=None):
    """Capture checkerboard images using RealSense D415."""
    if save_dir is None:
        save_dir = CALIBRATION_DIR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_images = glob.glob(os.path.join(save_dir, '*.jpg'))
    for image_path in existing_images:
        os.remove(image_path)

    if existing_images:
        print(f"[INFO] Cleared {len(existing_images)} existing calibration image(s)")

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense camera: {e}")
        return

    img_count = 0
    print("[INFO] Capture calibration images")
    print("[INFO] Press 'q' to finish capturing")

    while True:
        if _stop_requested:
            break
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        display_frame = frame.copy()

        cv2.putText(display_frame, f'Images captured: {img_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, 'SPACE: Capture | Q: Done', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Calibration Images", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            filename = os.path.join(save_dir, f'calib_{img_count:02d}.jpg')
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved calibration image {img_count}")
            img_count += 1

    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[INFO] Total images captured: {img_count}")
    print(f"[INFO] Images saved to: {save_dir}")


def calibrate_camera(checkerboard_size=CHECKERBOARD_SIZE, square_size=SQUARE_SIZE):
    """Calibrate camera using checkerboard images."""
    image_dir = CALIBRATION_DIR
    yaml_file = CAMERA_PARAMS_FILE
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if len(images) == 0:
        print(f"[ERROR] No calibration images found in {image_dir}")
        return None, None

    print(f"[INFO] Processing {len(images)} images for calibration...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            sub_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(sub_corners)
            cv2.drawChessboardCorners(img, checkerboard_size, sub_corners, ret)
            cv2.imshow('Calibration Progress', img)
            cv2.waitKey(100)
            print(f"[INFO] {os.path.basename(fname)}: ✓")
        else:
            print(f"[WARN] {os.path.basename(fname)}: ✗")

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("[ERROR] No valid checkerboard detections. Cannot calibrate.")
        return None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"[INFO] Calibration RMS error: {ret:.4f}")

    # Compute mean reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"[INFO] Mean reprojection error: {mean_error:.4f} pixels")

    data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.tolist(),
        'rms_error': float(ret),
        'mean_reprojection_error': float(mean_error),
        'image_width': int(gray.shape[1]),
        'image_height': int(gray.shape[0]),
        'num_images_used': len(objpoints)
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Camera parameters saved to {yaml_file}")

    return mtx, dist


def test_undistortion():
    """Test undistortion using saved calibration parameters."""
    yaml_file = CAMERA_PARAMS_FILE
    image_dir = CALIBRATION_DIR
    
    try:
        with open(yaml_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['distortion_coefficients'])
    except FileNotFoundError:
        print(f"[ERROR] {yaml_file} not found. Run calibration first.")
        return

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if len(images) == 0:
        print(f"[ERROR] No images found in {image_dir} to test undistortion.")
        return

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    comparison = np.hstack((cv2.resize(img, (640, 480)), cv2.resize(dst, (640, 480))))
    cv2.putText(comparison, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, 'Undistorted', (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Undistortion Test', comparison)
    print("[INFO] Press any key to close undistortion test window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[INFO] Undistortion test complete")


def franka_pick_and_place():
    """Franka pick and place operation."""
    print("\n[INFO] Franka Pick and Place Mode")
    print("[INFO] Please position the cube on the grid and ensure the camera can see the markers.")
    print("[INFO] The system will automatically detect the cube and execute the pick and place operation.")
    
    input("[INFO] Press Enter to start detection and proceed with pick and place...")
    print("[INFO] Starting cube detection and pick and place sequence...")
    print("[INFO] (This is handled by vision_bridge_node and pick_place_node)")
    print("[INFO] Monitor the ROS2 output for progress updates")


class UserInterfaceNode(Node):
    """ROS2 Node for user interface control."""
    
    def __init__(self):
        super().__init__('user_interface_node')
        self.get_logger().info('User Interface Node initialized')


def main():
    """Main menu loop."""
    print("\n" + "="*50)
    print("  Franka Pick and Place System")
    print("  ROS2 User Interface")
    print("="*50)
    
    rclpy.init()
    ui_node = UserInterfaceNode()
    
    try:
        while True:
            print("\n=== Main Menu ===")
            print("1. Capture calibration images")
            print("2. Calibrate camera")
            print("3. Test undistortion")
            print("4. Detect cube on ArUco grid")
            print("5. Franka Pick and Place")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()

            if choice == '1':
                print("\n[ACTION] Capturing calibration images...")
                capture_calibration_images()
                
            elif choice == '2':
                print("\n[ACTION] Calibrating camera...")
                calibrate_camera()
                print("[INFO] Camera calibration complete")
                
            elif choice == '3':
                print("\n[ACTION] Testing undistortion...")
                test_undistortion()
                
            elif choice == '4':
                print("\n[ACTION] Starting cube detection on ArUco grid...")
                print("[INFO] The vision_bridge_node is running and publishing cube positions")
                print("[INFO] Monitor /vision/cube_pose topic for cube position data")
                monitoring = input("Monitor detection output? (y/n): ").strip().lower()
                if monitoring == 'y':
                    print("[INFO] Monitoring vision output for 30 seconds...")
                    print("(Check ROS2 topic /vision/cube_pose with: ros2 topic echo /vision/cube_pose)")
                    time.sleep(30)
                    
            elif choice == '5':
                print("\n[ACTION] Franka Pick and Place...")
                franka_pick_and_place()
                print("[INFO] Pick and place sequence initiated")
                
            elif choice == '6':
                print("\n[INFO] Exiting system...")
                break
                
            else:
                print("[WARN] Invalid choice. Please enter 1-6.")
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        rclpy.shutdown()
        print("[INFO] ROS2 system shutdown complete")
        print("[INFO] Goodbye!")


if __name__ == '__main__':
    main()
