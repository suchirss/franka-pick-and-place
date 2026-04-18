"""
aruco_cube_tracker.py

Comprehensive ArUco cube detection and camera calibration script for Intel RealSense D415.

Features:
- Capture checkerboard images for camera calibration
- Calibrate the camera and save parameters
- Test undistortion on saved images
- Detect a 5x6 ArUco reference grid and track a cube with ArUco markers (IDs 70-75)
- Compute cube position relative to the top-left marker of the grid (ID 15)
- Real-time visualization with axes and marker IDs
- Save screenshots from the live feed

Requirements:
- Python 3.9+
- Packages: opencv-contrib-python, pyrealsense2, numpy, pyyaml
- Intel RealSense D415 camera connected
- Calibration checkerboard (recommended inner corners: 5x7, square size: 1.5 inch)

Usage:
    python aruco_cube_tracker.py
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import glob
import time
import signal
import sys
import time
from collections import deque
from cv_transform.warp_plane import WarpPlane

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty


# ------------------- MarkerTracker Class -------------------
class MarkerTracker:
    """Track ArUco marker 3D positions and compute velocities."""

    def __init__(self, max_path_length=100):
        self.paths = {}         # marker_id -> deque of 3D positions
        self.max_path_length = max_path_length
        self.velocities = {}    # marker_id -> (vx, vy, vz)
        self.last_positions = {}  # marker_id -> last 3D position
        self.last_time = {}       # marker_id -> last timestamp

    def update_position(self, marker_id, pos_3d, timestamp):
        """Update marker 3D position and compute instantaneous velocity."""
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
        """Return the most recent 3D position."""
        if marker_id in self.paths and len(self.paths[marker_id]) > 0:
            return self.paths[marker_id][-1]
        return None


# ------------------- CubeCenterTracker Class -------------------
class CubeCenterTracker:
    """
    Track the 3D center of a cube using detected ArUco markers.
    
    Assumes cube markers (IDs 70-75) are placed at known offsets from the cube center.
    The marker offsets assume a 1.5-inch cube with markers on top and sides.
    """
    
    # Marker ID to assumed position offset from cube center (in meters, assuming markers at ±0.75 inches = ±0.01905m)
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
        
        Strategy: Use marker positions and their known offsets to estimate cube center.
        For multiple markers, we solve for the cube center that best fits all detections.
        
        Args:
            origin_pos: 3D position of the grid origin (marker ID 15)
            
        Returns:
            Cube center position relative to origin (or None if no markers detected)
        """
        if not self.detected_markers:
            return None
        
        # Simple approach: average the back-calculated centers from each marker
        estimated_centers = []
        for marker_id, tvec in self.detected_markers.items():
            offset = self.MARKER_OFFSETS[marker_id]
            # Cube center = marker position - marker offset
            estimated_center = tvec - offset
            estimated_centers.append(estimated_center)
        
        if estimated_centers:
            # Average all estimated centers to reduce error
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
        
        # Ensure proper shape for cv2.projectPoints
        points_3d = np.array([cube_center_3d.flatten()], dtype=np.float32)
        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
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
    
    def get_cube_grid_cell(self, warp_manager, camera_matrix, dist_coeffs, origin_pos):
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
        # Calculate cube center in 3D
        cube_center_3d = self.calculate_cube_center_3d(origin_pos)
        if cube_center_3d is None:
            return None
        
        # Project to 2D pixel space
        pixel_pos = self.project_cube_center_to_pixel(cube_center_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        # Get grid cell from pixel position
        grid_cell = warp_manager.pixel_to_grid_cell(pixel_pos[0], pixel_pos[1])
        return grid_cell
    
    def get_cube_grid_position(self, warp_manager, camera_matrix, dist_coeffs, origin_pos):
        """
        Get the continuous grid coordinates of the cube's bottom center.
        
        Args:
            warp_manager: WarpPlane instance for grid mapping
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            origin_pos: 3D position of grid origin
            
        Returns:
            (grid_row, grid_col) continuous coordinates or None if calculation fails
        """
        # Calculate cube center in 3D
        cube_center_3d = self.calculate_cube_center_3d(origin_pos)
        if cube_center_3d is None:
            return None
        
        # Project to 2D pixel space
        pixel_pos = self.project_cube_center_to_pixel(cube_center_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        # Get grid position from pixel
        grid_pos = warp_manager.pixel_to_grid(pixel_pos[0], pixel_pos[1])
        return grid_pos
    
    def get_cube_bottom_3d(self, origin_pos):
        """
        Get the 3D position of the cube's bottom center (vertical center offset downward by 0.75 inches).
        
        In world coordinates, "downward" means negative Z direction.
        Offset is 0.75 inches (0.01905 m).
        
        Args:
            origin_pos: 3D position of grid origin
            
        Returns:
            3D position of cube bottom or None if no markers detected
        """
        # Get cube center
        cube_center_3d = self.calculate_cube_center_3d(origin_pos)
        if cube_center_3d is None:
            return None
        
        # Offset downward by 0.75 inches (half cube height) in negative Z direction
        CUBE_HEIGHT_OFFSET = 0.01905  # 0.75 inches in meters
        cube_bottom_3d = cube_center_3d.copy()
        cube_bottom_3d[2] -= CUBE_HEIGHT_OFFSET  # Move down (negative Z)
        
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
        # Get cube bottom in 3D
        cube_bottom_3d = self.get_cube_bottom_3d(origin_pos)
        if cube_bottom_3d is None:
            return None
        
        # Project to 2D pixel space
        pixel_pos = self.project_cube_center_to_pixel(cube_bottom_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        # Get grid cell from pixel position
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
            (grid_row, grid_col) continuous coordinates or None if calculation fails
        """
        # Get cube bottom in 3D
        cube_bottom_3d = self.get_cube_bottom_3d(origin_pos)
        if cube_bottom_3d is None:
            return None
        
        # Project to 2D pixel space
        pixel_pos = self.project_cube_center_to_pixel(cube_bottom_3d, camera_matrix, dist_coeffs)
        if pixel_pos is None:
            return None
        
        # Get grid position from pixel
        grid_pos = warp_manager.pixel_to_grid(pixel_pos[0], pixel_pos[1])
        return grid_pos

# Global flag for graceful exit
_stop_requested = False

# Default parameters
CHECKERBOARD_SIZE = (5, 7)      # inner corners for calibration checkerboard
SQUARE_SIZE = 0.0381            # 1.5 inch in meters
ARUCO_DICT = cv2.aruco.DICT_4X4_250
GRID_MARKER_IDS = list(range(15, 45))
CUBE_MARKER_IDS = list(range(70, 76))
MARKER_SIZE_M = 0.0381           # 1.5 inch markers

# --------------------------- Signal Handling ---------------------------
def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_sigint)

# --------------------------- Camera Calibration ---------------------------
def capture_calibration_images(save_dir='aruco\\calibration_images'):
    """Capture checkerboard images using RealSense D415."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_images = glob.glob(os.path.join(save_dir, '*.jpg'))
    for image_path in existing_images:
        os.remove(image_path)

    if existing_images:
        print(f"[INFO] Cleared {len(existing_images)} existing calibration image(s) from {save_dir}")

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense camera: {e}")
        return

    img_count = 0
    print("Press SPACE to capture an image, ESC to quit.")

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
        cv2.putText(display_frame, 'SPACE: Capture | ESC: Quit', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Calibration Images", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            filename = os.path.join(save_dir, f'calib_{img_count:02d}.jpg')
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")
            img_count += 1

    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[INFO] Total images captured: {img_count}")

def calibrate_camera(checkerboard_size=CHECKERBOARD_SIZE, square_size=SQUARE_SIZE,
                     image_dir='aruco\\calibration_images', yaml_file='camera_params.yaml'):
    """Calibrate camera using checkerboard images."""
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
            cv2.imshow('Calibration', img)
            cv2.waitKey(100)
            print(f"[INFO] {fname}: ✓")
        else:
            print(f"[WARN] {fname}: ✗")

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

def test_undistortion(yaml_file='camera_params.yaml', image_dir='aruco\\calibration_images'):
    """Test undistortion using saved calibration parameters."""
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('undistortion_test.jpg', comparison)
    print("[INFO] Undistortion comparison saved as 'undistortion_test.jpg'")

# --------------------------- ArUco Detection ---------------------------
def load_camera_params(yaml_file='camera_params.yaml'):
    """Load camera matrix and distortion coefficients from YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['distortion_coefficients'])
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"[ERROR] {yaml_file} not found! Run calibration first.")
        return None, None

def estimate_pose_single_markers(corners, marker_size, camera_matrix, dist_coeffs):
    """Estimate pose for single markers."""
    marker_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corner in corners:
        retval, rvec, tvec = cv2.solvePnP(marker_points, corner.astype(np.float32),
                                          camera_matrix, dist_coeffs,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if retval:
            rvecs.append(rvec)
            tvecs.append(tvec)
    return np.array(rvecs), np.array(tvecs), marker_points

def detect_aruco_grid_and_cube(marker_size_m=MARKER_SIZE_M):
    """Detect ArUco grid and cube markers using RealSense D415."""
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None:
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    tracker = MarkerTracker(max_path_length=100)
    cube_tracker = CubeCenterTracker()
    warp_manager = WarpPlane()
    show_grid = True
    show_warped = False

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Could not start RealSense camera: {e}")
        warp_manager.destroy_warp_plane_instance()
        return

    print("[INFO] Press 'q' to quit, 's' to save screenshot.")
    print("[INFO] Press 'g' to toggle grid overlay, 'w' to toggle warped view.")
    screenshot_count = 0

    while True:
        if _stop_requested:
            break
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(corners, marker_size_m,
                                                           camera_matrix, dist_coeffs)

            if not warp_manager.is_calibrated:
                if warp_manager.compute_homography(corners, ids):
                    print("[INFO] Grid homography completed successfully")
                else:
                    print("[WARN] Not enough grid markers visible for homography")

            # Find top-left grid marker (ID 15) to define origin
            origin_idx = None
            if 15 in ids:
                origin_idx = np.where(ids.flatten() == 15)[0][0]
                origin_pos = tvecs[origin_idx]

            # Update cube center tracker with detected cube markers
            cube_tracker.update_detected_markers(ids, tvecs)

            for i, marker_id in enumerate(ids.flatten()):
                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], marker_size_m*0.5)
                # Compute relative position to origin if cube marker
                if marker_id in CUBE_MARKER_IDS and origin_idx is not None:
                    rel_pos = (tvecs[i] - origin_pos).flatten()
                    tracker.update_position(marker_id, rel_pos, time.time())
                    text = f"ID:{marker_id} Rel(m): [{rel_pos[0]:.2f}, {rel_pos[1]:.2f}, {rel_pos[2]:.2f}]"
                    corner_center = corners[i][0].mean(axis=0).astype(int)
                    cv2.putText(frame, text, tuple(corner_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    distance = np.linalg.norm(rel_pos)

            # Calculate and display cube bottom grid position
            if warp_manager.is_calibrated and origin_idx is not None and len(cube_tracker.detected_markers) > 0:
                # Get cube bottom grid cell
                cube_grid_cell = cube_tracker.get_cube_bottom_grid_cell(warp_manager, camera_matrix, 
                                                                   dist_coeffs, origin_pos)
                cube_grid_pos = cube_tracker.get_cube_bottom_grid_position(warp_manager, camera_matrix,
                                                                     dist_coeffs, origin_pos)
                
                if cube_grid_cell and cube_grid_pos:
                    cube_grid_text = f"CUBE BOTTOM - Cell: ({cube_grid_cell[0]}, {cube_grid_cell[1]}) | " \
                                    f"Pos: ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f})"
                    cv2.putText(frame, cube_grid_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
                    
                    # Also draw the projected cube bottom on the frame
                    if cube_tracker.last_cube_center_pixel:
                        px = cube_tracker.last_cube_center_pixel.astype(int)
                        cv2.circle(frame, tuple(px), 8, (0, 0, 255), 2)  # Red circle for cube bottom
                        cv2.putText(frame, "CUBE BOTTOM", (px[0]-40, px[1]-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    print(f"CUBE BOTTOM - Grid cell ({cube_grid_cell[0]}, {cube_grid_cell[1]}), " 
                          f"continuous ({cube_grid_pos[0]:.2f}, {cube_grid_pos[1]:.2f})")

            if show_grid and warp_manager.is_calibrated:
                warp_manager.draw_grid_overlay(frame)

        else:
            cv2.putText(frame, 'No markers detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ------------------- Print 3D positions for cubes -------------------
        for mid in CUBE_MARKER_IDS:
            pos = tracker.get_latest_position(mid)
            if pos is not None:
                print(f"Cube {mid}: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} m")

        cv2.imshow("ArUco Cube Tracking", frame)

        if show_warped and warp_manager.is_calibrated:
            warped = warp_manager.warp_frame(frame)
            if warped is not None:
                cv2.imshow("Grid Normalized, Top Down View", warped)
            elif not show_warped:
                try:
                    cv2.destroyWindow("Grid Normalized, Top Down View")
                except cv2.error:
                    pass


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'screenshot_{screenshot_count:02d}.jpg'
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")
            screenshot_count += 1

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
        elif key == ord('w'):
            show_warped = not show_warped
            print(f"Warped view: {'ON' if show_warped else 'OFF'}")
        elif key == ord('r'):
            if ids is not None and len(ids) > 0:
                warp_manager._homography = None
                warp_manager._inverse_homography = None
                if warp_manager.compute_homography(corners, ids):
                    print("[INFO] Homography has been recalibrated")
                else:
                    print("[WARN] Recalibration failed: not enough grid markers")

    pipeline.stop()
    warp_manager.destroy_warp_plane_instance()
    cv2.destroyAllWindows()

import os
import sys
import time

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty


def _get_key_nonblocking():
    """
    Return a single pressed key as lowercase text, or None if no key is ready.
    Works in the terminal without OpenCV.
    """
    if os.name == "nt":
        if msvcrt.kbhit():
            ch = msvcrt.getch()

            # Ignore special multi-byte keys like arrows/function keys
            if ch in (b'\x00', b'\xe0'):
                if msvcrt.kbhit():
                    msvcrt.getch()
                return None

            try:
                return ch.decode("utf-8", errors="ignore").lower()
            except Exception:
                return None
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            ch = sys.stdin.read(1)
            return ch.lower()
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def calibrate_franka_origin():
    """Terminal-based manual control for calibrating Franka origin."""
    print("[INFO] Franka manual control started.")
    print("[INFO]   w = move up")
    print("[INFO]   a = move left")
    print("[INFO]   s = move down")
    print("[INFO]   d = move right")
    print("[INFO]   r = move out")
    print("[INFO]   f = move in")
    print("[INFO]   c = calibrate origin")
    print("[INFO]   q = return to main menu")

    movement_messages = {
        'w': "[INFO] Moving Franka up...",
        'a': "[INFO] Moving Franka left...",
        's': "[INFO] Moving Franka down...",
        'd': "[INFO] Moving Franka right...",
        'r': "[INFO] Moving Franka out...",
        'f': "[INFO] Moving Franka in...",
    }

    try:
        while True:
            key = _get_key_nonblocking()

            if key is None:
                time.sleep(0.01)
                continue

            if key in movement_messages:
                print(movement_messages[key])

            elif key == 'c':
                print("[INFO] Calibrating Franka origin...")
                break

            elif key == 'q':
                print("[INFO] Returning to main menu...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Returning to main menu...")

def franka_pick_and_place():
    """Franka pick and place operation."""
    print("[INFO] Input a grid cell to place the cube")
    row = input("Enter cell row: ").strip().upper()
    col = input("Enter cell column: ").strip().upper()
    print(f"[INFO] Grid cell selection ({row}, {col})...")
    choice = input("Confirm pick and place operation? (y/n): ").strip().lower()
    if choice == 'y':
        print(f"[INFO] Executing pick and place to cell ({row}, {col})...")
    if choice == 'n':
        print("[INFO] Operation cancelled.")

# --------------------------- Main Menu ---------------------------
def main():
    while True:
        print("\n=== ArUco Cube Tracker Menu ===")
        print("1. Capture calibration images")
        print("2. Calibrate camera")
        print("3. Test undistortion")
        print("4. Detect cube on ArUco grid (live feed)")
        print("5. Calibrate Franka origin")
        print("6. Franka Pick and Place")
        print("7. Exit")
        choice = input("Enter choice (1-7): ")

        if choice == '1':
            capture_calibration_images()
        elif choice == '2':
            calibrate_camera()
        elif choice == '3':
            test_undistortion()
        elif choice == '4':
            detect_aruco_grid_and_cube()
        elif choice == '5':
            calibrate_franka_origin()
        elif choice == '6':
            franka_pick_and_place()
        elif choice == '7':
            print("[INFO] Exiting...")
            break
        else:
            print("[WARN] Invalid choice. Please enter 1-7.")

if __name__ == '__main__':
    main()