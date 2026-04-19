#!/usr/bin/env python3

import time
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import os
import sys
from ament_index_python.packages import get_package_share_directory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(PKG_DIR, 'config')
CAMERA_PARAMS_FILE = os.path.join(CONFIG_DIR, 'camera_params.yaml')
sys.path.append(PKG_DIR)

from cv_transform.warp_plane import WarpPlane

ARUCO_DICT = cv2.aruco.DICT_4X4_250
GRID_MARKER_ID_ORIGIN = 15
CUBE_MARKER_IDS = list(range(70, 76))
MARKER_SIZE_M = 0.0381  # 1.5 inch in meters

_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_sigint)


def load_camera_params(yaml_file='camera_params.yaml'):
    yaml_path = CAMERA_PARAMS_FILE

    try:
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        camera_matrix = np.array(calib_data['camera_matrix']).reshape((3, 3))
        dist_coeffs = np.array(calib_data['distortion_coefficients']).reshape((-1, 1))
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"[ERROR] {yaml_path} not found! Run calibration first.")
        return None, None


def estimate_pose_single_markers(corners, marker_size, camera_matrix, dist_coeffs):
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

    return np.array(rvecs), np.array(tvecs)


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


class VisionBridgeNode(Node):
    def __init__(self):
        super().__init__('vision_bridge_node')

        self.publisher_ = self.create_publisher(PoseStamped, '/vision/cube_pose', 10)

        self.camera_matrix, self.dist_coeffs = load_camera_params('camera_params.yaml')

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
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