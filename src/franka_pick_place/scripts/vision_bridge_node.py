#!/usr/bin/env python3

"""
Vision Bridge Node - Central vision and camera processing module

Contains all ArUco marker detection, cube tracking, and camera calibration functionality.
Used by both the ROS2 autonomous node (VisionBridgeNode) and the interactive menu (user_interface).
"""

import time
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import sys
import glob
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PKG_DIR)

from cv_transform.warp_plane import WarpPlane

ARUCO_DICT = cv2.aruco.DICT_4X4_250
GRID_MARKER_ID_ORIGIN = 15
CUBE_MARKER_IDS = list(range(70, 76))
GRID_MARKER_IDS = list(range(15, 45))
MARKER_SIZE_M = 0.0381  # 1.5 inch in meters
CHECKERBOARD_SIZE = (5, 7)
SQUARE_SIZE = 0.0381

_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_sigint)


def get_config_paths():
    """Get standard config paths based on script location."""
    config_dir = os.path.join(PKG_DIR, 'config')
    calibration_dir = os.path.join(config_dir, 'calibration_images')
    camera_params_file = os.path.join(config_dir, 'camera_params.yaml')
    
    return {
        'script_dir': SCRIPT_DIR,
        'pkg_dir': PKG_DIR,
        'config_dir': config_dir,
        'calibration_dir': calibration_dir,
        'camera_params_file': camera_params_file,
    }


def load_camera_params(yaml_file='camera_params.yaml'):
    """Load camera matrix and distortion coefficients from YAML file."""
    paths = get_config_paths()
    yaml_path = paths['camera_params_file']
    
    try:
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float64)
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"[ERROR] {yaml_path} not found! Run calibration first.")
        return None, None


def estimate_pose_single_markers(corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate pose for single markers using solvePnP.
    Uses older OpenCV API for compatibility with older ROS2 distributions.
    
    Args:
        corners: List of marker corner arrays
        marker_size: Physical size of marker in meters
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        
    Returns:
        Tuple of (rvecs array, tvecs array, marker_points)
    """
    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corner in corners:
        retval, rvec, tvec = cv2.solvePnP(
            marker_points,
            corner.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if retval:
            rvecs.append(rvec)
            tvecs.append(tvec)

    return np.array(rvecs), np.array(tvecs), marker_points


class MarkerTracker:
    """Track ArUco marker 3D positions and compute velocities."""

    def __init__(self, max_path_length=100):
        """
        Initialize marker tracker.
        
        Args:
            max_path_length: Maximum number of position samples to store per marker
        """
        self.paths = {}           # marker_id -> deque of 3D positions
        self.max_path_length = max_path_length
        self.velocities = {}      # marker_id -> (vx, vy, vz)
        self.last_positions = {}  # marker_id -> last 3D position
        self.last_time = {}       # marker_id -> last timestamp

    def update_position(self, marker_id, pos_3d, timestamp):
        """
        Update marker 3D position and compute instantaneous velocity.
        
        Args:
            marker_id: ID of the marker
            pos_3d: 3D position array (x, y, z)
            timestamp: Current timestamp for velocity computation
        """
        if marker_id not in self.paths:
            self.paths[marker_id] = deque(maxlen=self.max_path_length)
            self.velocities[marker_id] = (0.0, 0.0, 0.0)
            self.last_positions[marker_id] = pos_3d
            self.last_time[marker_id] = timestamp

        self.paths[marker_id].append(pos_3d)

        last_pos = self.last_positions[marker_id]
        dt = timestamp - self.last_time[marker_id]

        if dt > 0:
            vx, vy, vz = [(pos_3d[i] - last_pos[i]) / dt for i in range(3)]
            self.velocities[marker_id] = (vx, vy, vz)

        self.last_positions[marker_id] = pos_3d
        self.last_time[marker_id] = timestamp

    def get_latest_position(self, marker_id):
        """
        Return the most recent 3D position for a marker.
        
        Args:
            marker_id: ID of the marker
            
        Returns:
            3D position array or None if no data
        """
        if marker_id in self.paths and len(self.paths[marker_id]) > 0:
            return self.paths[marker_id][-1]
        return None


class CubeCenterTracker:
    """
    Track the 3D center of a cube using detected ArUco markers.
    
    Assumes cube markers (IDs 70-75) are placed at known offsets from the cube center.
    The marker offsets assume a 1.5-inch cube with markers on top and sides.
    """
    
    # Marker ID to offset from cube center (in meters)
    # Adjust these based on your actual cube marker layout
    MARKER_OFFSETS = {
        70: np.array([ 0.01905,  0.01905,  0.01905]),  # Top-Right-Front
        71: np.array([-0.01905,  0.01905,  0.01905]),  # Top-Left-Front
        72: np.array([-0.01905,  0.01905, -0.01905]),  # Top-Left-Back
        73: np.array([ 0.01905,  0.01905, -0.01905]),  # Top-Right-Back
        74: np.array([ 0.01905, -0.01905,  0.01905]),  # Bottom-Right-Front
        75: np.array([-0.01905, -0.01905,  0.01905]),  # Bottom-Left-Front
    }
    
    def __init__(self):
        """Initialize the cube center tracker."""
        self.last_cube_center_3d = None
        self.last_cube_center_pixel = None
        self.detected_markers = {}  # marker_id -> tvec (3D position)
    
    def update_detected_markers(self, marker_ids, tvecs):
        """
        Update the set of detected cube markers and their 3D positions.
        
        Args:
            marker_ids: Array of detected marker IDs
            tvecs: Array of translation vectors (3D positions) for each marker
        """
        self.detected_markers.clear()
        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids.flatten()):
                if marker_id in self.MARKER_OFFSETS:
                    self.detected_markers[marker_id] = tvecs[i]
    
    def calculate_cube_center_3d(self, origin_pos):
        """
        Calculate the 3D position of the cube center from detected markers.
        
        Args:
            origin_pos: 3D position of the grid origin (marker ID 15)
            
        Returns:
            Cube center position relative to origin (or None if no markers detected)
        """
        if not self.detected_markers:
            return None
        
        # Average the back-calculated centers from each marker for robustness
        estimated_centers = []
        for marker_id, tvec in self.detected_markers.items():
            offset = self.MARKER_OFFSETS[marker_id]
            # Cube center = marker position - marker offset
            estimated_center = tvec - offset
            estimated_centers.append(estimated_center)
        
        if estimated_centers:
            cube_center = np.mean(estimated_centers, axis=0)
            self.last_cube_center_3d = cube_center
            
            # Return relative to origin
            relative_center = cube_center - origin_pos
            return relative_center
        
        return None
    
    def project_cube_center_to_pixel(self, cube_center_3d, camera_matrix, dist_coeffs):
        """
        Project the 3D cube center position onto the 2D image plane.
        
        Args:
            cube_center_3d: 3D position of cube center (1x3 or 3x1 array)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            (pixel_x, pixel_y) tuple or None if projection fails
        """
        if cube_center_3d is None:
            return None
        
        points_3d = np.array([cube_center_3d.flatten()], dtype=np.float64)
        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        try:
            projected_points, _ = cv2.projectPoints(
                points_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            pixel_pos = projected_points[0][0]
            self.last_cube_center_pixel = pixel_pos
            return float(pixel_pos[0]), float(pixel_pos[1])
        except Exception as e:
            print(f"[ERROR] Failed to project cube center: {e}")
            return None
    
    def get_cube_bottom_3d(self, origin_pos):
        """
        Get the 3D position of the cube's bottom center.
        
        Cube bottom is defined as the cube center offset downward by 0.75 inches
        (half the cube height) in the negative Z direction.
        
        Args:
            origin_pos: 3D position of grid origin
            
        Returns:
            3D position of cube bottom or None if no markers detected
        """
        cube_center_3d = self.calculate_cube_center_3d(origin_pos)
        if cube_center_3d is None:
            return None
        
        # Offset downward by 0.75 inches (0.01905 meters) in negative Z
        CUBE_HEIGHT_OFFSET = 0.01905
        cube_bottom_3d = cube_center_3d.copy()
        cube_bottom_3d[2] -= CUBE_HEIGHT_OFFSET
        
        return cube_bottom_3d
    
    def get_cube_bottom_grid_cell(self, warp_manager, camera_matrix, dist_coeffs, origin_pos):
        """
        Get the grid cell containing the cube's bottom center.
        
        Args:
            warp_manager: WarpPlane instance for grid mapping
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            origin_pos: 3D position of grid origin
            
        Returns:
            (grid_row, grid_col) tuple or None if calculation fails
        """
        cube_bottom_3d = self.get_cube_bottom_3d(origin_pos)
        if cube_bottom_3d is None:
            return None
        
        pixel_pos = self.project_cube_center_to_pixel(cube_bottom_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        grid_cell = warp_manager.pixel_to_grid_cell(pixel_pos[0], pixel_pos[1])
        return grid_cell
    
    def get_cube_bottom_grid_position(self, warp_manager, camera_matrix, dist_coeffs, origin_pos):
        """
        Get the continuous grid coordinates of the cube's bottom center.
        
        Args:
            warp_manager: WarpPlane instance for grid mapping
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            origin_pos: 3D position of grid origin
            
        Returns:
            (grid_x, grid_y) continuous coordinates or None if calculation fails
        """
        cube_bottom_3d = self.get_cube_bottom_3d(origin_pos)
        if cube_bottom_3d is None:
            return None
        
        pixel_pos = self.project_cube_center_to_pixel(cube_bottom_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        grid_pos = warp_manager.pixel_to_grid(pixel_pos[0], pixel_pos[1])
        return grid_pos


# ===== CALIBRATION FUNCTIONS =====

def capture_calibration_images(save_dir=None):
    """Capture checkerboard images using RealSense D415."""
    if save_dir is None:
        paths = get_config_paths()
        save_dir = paths['calibration_dir']
    
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
    print("[INFO] Press 'SPACE' to capture, 'Q' to finish")

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
    paths = get_config_paths()
    image_dir = paths['calibration_dir']
    yaml_file = paths['camera_params_file']
    
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
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if len(objpoints) == 0:
        print("[ERROR] Could not find checkerboard corners in any image")
        return None, None

    print(f"[INFO] Found checkerboard in {len(objpoints)} images")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if not ret:
        print("[ERROR] Camera calibration failed")
        return None, None

    calib_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.flatten().tolist(),
    }

    try:
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        with open(yaml_file, 'w') as f:
            yaml.dump(calib_data, f)
        print(f"[INFO] Camera calibration saved to {yaml_file}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"[ERROR] Could not save calibration: {e}")
        return None, None


def test_undistortion():
    """Test camera calibration by displaying undistorted frames."""
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None:
        print("[ERROR] Camera calibration not found. Run calibration first.")
        return

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense camera: {e}")
        return

    print("[INFO] Displaying calibrated and undistorted frames")
    print("[INFO] Press 'Q' to exit")

    while True:
        if _stop_requested:
            break

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        cv2.imshow("Original", frame)
        cv2.imshow("Undistorted", undistorted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Undistortion test complete")


def detect_aruco_grid_and_cube(marker_size_m=MARKER_SIZE_M):
    """Detect ArUco grid and cube markers using RealSense D415."""
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None:
        return

    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters_create()
    tracker = MarkerTracker(max_path_length=100)
    cube_tracker = CubeCenterTracker()
    warp_manager = WarpPlane()
    show_grid = True

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense camera: {e}")
        warp_manager.destroy_warp_plane_instance()
        return

    print("[INFO] Press 'Q' to quit detection, 'G' to toggle grid")
    print("[INFO] Detecting ArUco markers and cube position...")

    while True:
        if _stop_requested:
            break
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, marker_size_m,
                                                           camera_matrix, dist_coeffs)

            if not warp_manager.is_calibrated:
                if warp_manager.compute_homography(corners, ids):
                    print("[INFO] Grid homography completed successfully")
                else:
                    print("[WARN] Not enough grid markers visible for homography")

            # Find grid origin marker (ID 15)
            ids_flat = ids.flatten()
            origin_idx = None
            if GRID_MARKER_ID_ORIGIN in ids_flat:
                origin_idx = np.where(ids_flat == GRID_MARKER_ID_ORIGIN)[0][0]
                origin_pos = tvecs[origin_idx]

                # Update cube tracker
                cube_tracker.update_detected_markers(ids, tvecs)

                for i, marker_id in enumerate(ids_flat):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      rvecs[i], tvecs[i], marker_size_m*0.5)

                # Display cube bottom grid position
                if warp_manager.is_calibrated and len(cube_tracker.detected_markers) > 0:
                    cube_grid_cell = cube_tracker.get_cube_bottom_grid_cell(warp_manager, camera_matrix, 
                                                                       dist_coeffs, origin_pos)
                    cube_grid_pos = cube_tracker.get_cube_bottom_grid_position(warp_manager, camera_matrix,
                                                                         dist_coeffs, origin_pos)
                    
                    if cube_grid_cell and cube_grid_pos:
                        cube_grid_text = f"CUBE BOTTOM - Cell: ({cube_grid_cell[0]}, {cube_grid_cell[1]}) | " \
                                        f"Pos: ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f})"
                        cv2.putText(frame, cube_grid_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
                        
                        if cube_tracker.last_cube_center_pixel is not None:
                            px = cube_tracker.last_cube_center_pixel.astype(int)
                            cv2.circle(frame, tuple(px), 8, (0, 0, 255), 2)
                            cv2.putText(frame, "CUBE BOTTOM", (px[0]-40, px[1]-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        print(f"CUBE BOTTOM - Grid cell ({cube_grid_cell[0]}, {cube_grid_cell[1]}), " 
                              f"continuous ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f})")

            if show_grid and warp_manager.is_calibrated:
                warp_manager.draw_grid_overlay(frame)

        else:
            cv2.putText(frame, 'No markers detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("ArUco Cube Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")

    pipeline.stop()
    warp_manager.destroy_warp_plane_instance()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete")



class VisionBridgeNode(Node):
    def __init__(self):
        super().__init__('vision_bridge_node')

        self.publisher_ = self.create_publisher(PoseStamped, '/vision/cube_pose', 10)

        self.camera_matrix, self.dist_coeffs = load_camera_params('camera_params.yaml')

        self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        self.warp_manager = WarpPlane()
        self.cube_tracker = CubeCenterTracker()
        self.last_pub_time = 0.0
        self.publish_period = 0.2  # 5 Hz max

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.timer = self.create_timer(0.03, self.process_frame)
        self.get_logger().info('Vision bridge started. Publishing /vision/cube_pose')

    def process_frame(self):
        global _stop_requested
        if _stop_requested:
            self.cleanup()
            return

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.detector_params
        )

        if ids is None or len(ids) == 0:
            cv2.putText(frame, 'No markers detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Vision Bridge", frame)
            cv2.waitKey(1)
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs = estimate_pose_single_markers(
            corners,
            MARKER_SIZE_M,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not self.warp_manager.is_calibrated:
            self.warp_manager.compute_homography(corners, ids)

        ids_flat = ids.flatten()

        origin_idx = None
        if GRID_MARKER_ID_ORIGIN in ids_flat:
            origin_idx = np.where(ids_flat == GRID_MARKER_ID_ORIGIN)[0][0]
        else:
            cv2.putText(frame, 'Origin marker 15 not visible', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Vision Bridge", frame)
            cv2.waitKey(1)
            return

        origin_pos = tvecs[origin_idx]

        # Update cube center tracker with all detected markers
        self.cube_tracker.update_detected_markers(ids, tvecs)

        # Calculate cube bottom grid position
        cube_grid_cell = None
        cube_grid_pos = None
        if len(self.cube_tracker.detected_markers) > 0 and self.warp_manager.is_calibrated:
            cube_grid_cell = self.cube_tracker.get_cube_bottom_grid_cell(
                self.warp_manager, self.camera_matrix, self.dist_coeffs, origin_pos
            )
            cube_grid_pos = self.cube_tracker.get_cube_bottom_grid_position(
                self.warp_manager, self.camera_matrix, self.dist_coeffs, origin_pos
            )

        # For publishing, use first visible cube marker (for backward compatibility)
        chosen_idx = None
        chosen_id = None
        for i, marker_id in enumerate(ids_flat):
            if marker_id in CUBE_MARKER_IDS:
                chosen_idx = i
                chosen_id = marker_id
                break

        if chosen_idx is None:
            cv2.putText(frame, 'No cube marker visible', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Vision Bridge", frame)
            cv2.waitKey(1)
            return

        rel_pos = (tvecs[chosen_idx] - origin_pos).flatten()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'aruco_marker_15'
        msg.pose.position.x = float(rel_pos[0])
        msg.pose.position.y = float(rel_pos[1])
        msg.pose.position.z = float(rel_pos[2])

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        now = time.time()
        if now - self.last_pub_time >= self.publish_period:
            self.publisher_.publish(msg)
            self.last_pub_time = now
            
            # Log cube bottom grid information
            if cube_grid_cell and cube_grid_pos:
                self.get_logger().info(
                    f'Cube bottom: Cell ({cube_grid_cell[0]}, {cube_grid_cell[1]}), '
                    f'Pos ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f}) | '
                    f'Marker {chosen_id}: x={rel_pos[0]:.3f}, y={rel_pos[1]:.3f}, z={rel_pos[2]:.3f}'
                )
            else:
                self.get_logger().info(
                    f'Published cube marker {chosen_id}: '
                    f'x={rel_pos[0]:.3f}, y={rel_pos[1]:.3f}, z={rel_pos[2]:.3f}'
                )

        # Draw marker info on frame
        center_px = corners[chosen_idx][0].mean(axis=0).astype(int)
        text = f'ID:{chosen_id} rel=({rel_pos[0]:.3f},{rel_pos[1]:.3f},{rel_pos[2]:.3f})'
        cv2.putText(frame, text, tuple(center_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw cube bottom grid information
        if cube_grid_cell and cube_grid_pos:
            cube_grid_text = f"CUBE BOTTOM - Cell: ({cube_grid_cell[0]}, {cube_grid_cell[1]}) | " \
                            f"Pos: ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f})"
            cv2.putText(frame, cube_grid_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            # Draw the projected cube bottom on the frame
            if self.cube_tracker.last_cube_center_pixel is not None:
                px = self.cube_tracker.last_cube_center_pixel.astype(int)
                cv2.circle(frame, tuple(px), 8, (0, 0, 255), 2)  # Red circle for cube bottom
                cv2.putText(frame, "CUBE BOTTOM", (px[0]-40, px[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.warp_manager.is_calibrated:
            self.warp_manager.draw_grid_overlay(frame)

        cv2.imshow("Vision Bridge", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup()

    def cleanup(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.warp_manager.destroy_warp_plane_instance()
        cv2.destroyAllWindows()
        self.get_logger().info('Vision bridge shutting down.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = VisionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()


if __name__ == '__main__':
    main()