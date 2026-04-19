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
    pkg_share = get_package_share_directory('franka_pick_place')
    yaml_path = os.path.join(pkg_share, 'config', yaml_file)

    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)

    camera_matrix = np.array(calib_data['camera_matrix']).reshape((3, 3))
    dist_coeffs = np.array(calib_data['distortion_coefficients']).reshape((-1, 1))
    return camera_matrix, dist_coeffs


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


class VisionBridgeNode(Node):
    def __init__(self):
        super().__init__('vision_bridge_node')

        self.publisher_ = self.create_publisher(PoseStamped, '/vision/cube_pose', 10)

        self.camera_matrix, self.dist_coeffs = load_camera_params('camera_params.yaml')

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        self.warp_manager = WarpPlane()
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
            self.get_logger().info(
                f'Published cube marker {chosen_id}: '
                f'x={rel_pos[0]:.3f}, y={rel_pos[1]:.3f}, z={rel_pos[2]:.3f}'
            )

        center_px = corners[chosen_idx][0].mean(axis=0).astype(int)
        text = f'ID:{chosen_id} rel=({rel_pos[0]:.3f},{rel_pos[1]:.3f},{rel_pos[2]:.3f})'
        cv2.putText(frame, text, tuple(center_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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