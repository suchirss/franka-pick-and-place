#!/usr/bin/env python3

"""
ROS2 User Interface for Franka Pick and Place System

Menu-driven interface that calls vision and calibration functions from vision_bridge_node.
Provides options for:
- Camera calibration and testing
- Cube detection on ArUco grid
- Franka pick and place operations
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PKG_DIR)

import rclpy
from rclpy.node import Node

# Import all vision functions and classes from vision_bridge_node
from vision_bridge_node import (
    capture_calibration_images,
    calibrate_camera,
    test_undistortion,
    detect_aruco_grid_and_cube,
    get_config_paths,
)

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
                detect_aruco_grid_and_cube()
                print("[INFO] Detection session complete")
                    
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
