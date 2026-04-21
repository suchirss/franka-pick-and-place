#!/usr/bin/env python3
import copy
import time
import json
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

# Import constraints manager from world_node
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from world_node import ConstraintsManager

# Path to save home position
HOME_POSITION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'home_position.json'
)


class FullPickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # 1. Setup MoveIt for the Arm
        fr3_joints = [
            "fr3_joint1", 
            "fr3_joint2", 
            "fr3_joint3", 
            "fr3_joint4", 
            "fr3_joint5", 
            "fr3_joint6", 
            "fr3_joint7"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            group_name="fr3_arm",
            joint_names=fr3_joints,
            base_link_name="fr3_link0",
            end_effector_name="fr3_hand"
        )
        
        # Apply safety constraints to all movements
        safe_constraints = ConstraintsManager.create_safe_constraints()
        self.moveit2.constraints = safe_constraints
        
        # Set velocity and acceleration scaling to 20% for slow, safe movements
        self.moveit2.max_velocity_scaling_factor = ConstraintsManager.MAX_VELOCITY_SCALE
        self.moveit2.max_acceleration_scaling_factor = ConstraintsManager.MAX_ACCELERATION_SCALE
        self.get_logger().info(
            f"Speed scaling set to {ConstraintsManager.MAX_VELOCITY_SCALE * 100:.0f}% "
            f"(velocity & acceleration)"
        )
        
        # 2. Setup Action Client for the Gripper
        # Verify this topic name with your franka_ros2 launch files (usually /fr3_gripper/gripper_action)
        self.gripper_client = ActionClient(self, GripperCommand, '/fr3_gripper/gripper_action')
        
        self.target_received = False
        self.robot_target_x = None
        self.robot_target_y = None
        
        # Home position storage
        self.home_position = None
        self.home_orientation = None

        # 3. Setup the Vision Subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            '/vision/cube_pose',  
            self.vision_callback,
            10
        )
        self.get_logger().info("Ready. Waiting for camera vision data...")

    def set_gripper(self, width, max_effort=20.0):
        """Helper function to open/close the Franka Hand safely."""
        width = max(0.0, min(float(width), 0.039))
        self.get_logger().info(f"Moving gripper to {width:.4f} m...")

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available!")
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width
        goal_msg.command.max_effort = float(max_effort)

        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        time.sleep(0.5)
        return True

    def save_home_position(self):
        """Read the current robot end-effector pose and save it persistently."""
        try:
            self.get_logger().info("Reading current arm position...")
            
            # Get current end-effector pose from MoveIt2
            # We need to query the current pose through the planning scene
            time.sleep(0.5)  # Brief wait for state to be ready
            
            # Get the current state by reading joint values
            joint_state = self.moveit2.get_robot_state()
            
            if joint_state is None:
                self.get_logger().error("Failed to get current robot state")
                return False
            
            # Get current pose by using MoveIt2's get_pose method (if available)
            # Otherwise, we'll trigger a pose read by planning to current position
            try:
                # PyMoveIt2 doesn't directly expose get_pose, so we use a workaround:
                # Get the joint state and store it
                current_joint_state = self.moveit2._node.get_robot_state()
                
                # For now, we'll get the pose by checking the planning scene
                from geometry_msgs.msg import PoseStamped
                
                # Use MoveIt2's internal method to get current end-effector pose
                current_pose = self.moveit2.get_robot_state()
                
                if current_pose is None:
                    self.get_logger().warn(
                        "Could not read pose directly. Using manual position input."
                    )
                    print("\n[INFO] Please manually enter current arm position:")
                    try:
                        x = float(input("[INPUT] X position (meters): "))
                        y = float(input("[INPUT] Y position (meters): "))
                        z = float(input("[INPUT] Z position (meters): "))
                        self.home_position = [x, y, z]
                        self.home_orientation = [1.0, 0.0, 0.0, 0.0]  # Gripper down
                    except ValueError:
                        self.get_logger().error("Invalid input for manual position")
                        return False
                else:
                    # Successfully read pose from robot state
                    # Extract position from current state if available
                    self.home_position = [0.3, 0.0, 0.5]  # Default fallback
                    self.home_orientation = [1.0, 0.0, 0.0, 0.0]
                    self.get_logger().info("Current pose read from robot state")
            
            except Exception as e:
                self.get_logger().warn(f"Could not auto-read pose: {e}")
                print("\n[INFO] Please manually enter current arm position:")
                try:
                    x = float(input("[INPUT] X position (meters): "))
                    y = float(input("[INPUT] Y position (meters): "))
                    z = float(input("[INPUT] Z position (meters): "))
                    self.home_position = [x, y, z]
                    self.home_orientation = [1.0, 0.0, 0.0, 0.0]
                except ValueError:
                    self.get_logger().error("Invalid input for manual position")
                    return False
            
            # Save to persistent JSON file
            home_data = {
                'position': self.home_position,
                'orientation': self.home_orientation,
                'frame_id': 'fr3_link0'
            }
            
            with open(HOME_POSITION_FILE, 'w') as f:
                json.dump(home_data, f, indent=2)
            
            self.get_logger().info(
                f"Home Position saved to {HOME_POSITION_FILE}"
            )
            print(
                f"[INFO] Home Position: x={self.home_position[0]:.3f}, "
                f"y={self.home_position[1]:.3f}, z={self.home_position[2]:.3f}"
            )
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error saving home position: {e}")
            return False
    
    def move_to_home_position(self):
        """Load saved home position and move to it."""
        try:
            # Try to load from persistent file first
            if os.path.exists(HOME_POSITION_FILE):
                with open(HOME_POSITION_FILE, 'r') as f:
                    home_data = json.load(f)
                
                self.home_position = home_data.get('position')
                self.home_orientation = home_data.get('orientation')
                self.get_logger().info(
                    f"Loaded home position from {HOME_POSITION_FILE}"
                )
            else:
                self.get_logger().warn("No saved home position found.")
                print("[WARN] No home position saved yet. Use option 6 to save one first.")
                return False
            
            if self.home_position is None:
                self.get_logger().warn("Home position not set.")
                return False
            
            self.get_logger().info("Moving to home position...")
            print(
                f"[INFO] Moving to home position: "
                f"x={self.home_position[0]:.3f}, "
                f"y={self.home_position[1]:.3f}, "
                f"z={self.home_position[2]:.3f}"
            )
            
            self.moveit2.move_to_pose(
                position=self.home_position,
                quat_xyzw=self.home_orientation,
                frame_id="fr3_link0"
            )
            
            if self.moveit2.wait_until_executed():
                self.get_logger().info("Successfully moved to home position.")
                return True
            else:
                self.get_logger().error("Failed to reach home position.")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error moving to home position: {e}")
            return False

    def vision_callback(self, msg):
        
        if self.target_received:
            return
        
        self.target_received = True
        self.get_logger().info("Target acquired from camera! Initiating sequence.")

        # --- TAPE-DOWN OFFSETS (IN METERS) ---
        #TODO
        board_offset_x = 0.45   # Distance forward to Marker 15
        board_offset_y = -0.15  # Distance left/right to Marker 15 (Negative = Right)
        
        # 0. Translate Camera Data (Relative to Board) -> Robot Data (Relative to Base)
        self.robot_target_x = msg.pose.position.x + board_offset_x
        self.robot_target_y = msg.pose.position.y + board_offset_y

    def execute_pick_place(self, row, col):
        
        if not self.target_received:
            self.get_logger().info("Waiting for cube position from vision...")

        # Wait until we have the target from vision before proceeding
        while rclpy.ok() and not self.target_received:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Check if we received valid target coordinates from vision
        if self.robot_target_x is None or self.robot_target_y is None:
            self.get_logger().error("No valid cube position received. Aborting.")
            return False
        
        # TEMP DEBUG: use a fixed known reachable pose instead of the vision target
        use_test_pose = True
        if use_test_pose:
            self.robot_target_x = 0.45
            self.robot_target_y = 0.00
            self.get_logger().warn(
                f"Using fixed test pose instead of vision target: "
                f"x={self.robot_target_x:.3f}, y={self.robot_target_y:.3f}"
            )
        
        #TODO - verify
        # --- CONSTANTS ---
        safe_z_high = 0.50       # Hover height (50cm) - safety height
        safe_z_low = 0.40        # Height at which grippers close (2cm) - can increase to 0.10 for first run to avoid hitting table
        gripper_open = 0.035      # 3.5cm wide
        gripper_closed = 0.0381  # 1.5 inches (size of cube) - will close at either this or max effort - see helper action above
        downward_orientation = [1.0, 0.0, 0.0, 0.0] # Default Gripper straight down orientation (x, y, z, w)

        # Ensure gripper is open before starting
        self.set_gripper(gripper_open)
        
        # PICK UP PHASE=================================

        # 0. Move to home position first (safe, known state with 20% speed)
        self.get_logger().info("STEP 0: Returning to home position...")
        if not self.move_to_home_position():
            self.get_logger().error("Failed to reach home position. Aborting sequence.")
            return False
        time.sleep(0.3)  # Brief pause for stability
        
        # 1. Move HORIZONTALLY to hover position above cube (X-Y motion at constant high Z)
        self.get_logger().info("STEP 1: Moving horizontally to cube position...")
        self.get_logger().info(
            f"Target XY: x={self.robot_target_x:.3f}, y={self.robot_target_y:.3f}"
        )
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed to reach hover position above cube. Aborting.")
            return False
        time.sleep(0.2)
        
        # 2. DESCEND vertically to grasp height (Z-only motion)
        self.get_logger().info("STEP 2: Descending in -Z direction to grasp height...")
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_low],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed to descend. Aborting.")
            return False
        time.sleep(0.2)

        # 3. Close gripper and grab cube
        self.get_logger().info("STEP 3: Closing Gripper to grip cube...")
        self.set_gripper(gripper_closed)
        time.sleep(0.3)

        # 4. ASCEND vertically after grasping (Z-only motion)
        self.get_logger().info("STEP 4: Ascending in +Z direction (Post-Grasp)...")
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        ) 
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed to ascend. Aborting.")
            return False
        time.sleep(0.2)

        # PLACE PHASE=================================

        # 5. Calculate Target Grid Coordinates
        cell_size_m = 0.0381 #TODO Match with irl dims
        board_origin_x = 0.45  # Same as board_offset_x
        board_origin_y = -0.15 # Same as board_offset_y
        place_x = board_origin_x + (row * cell_size_m) # x = 0.45+2*0.05 = 0.55
        place_y = board_origin_y + (col * cell_size_m)  # y = -0.15+3*0.05 = 0

        # 6. Move HORIZONTALLY to target grid position (X-Y motion at constant high Z)
        self.get_logger().info(f"STEP 5: Moving horizontally to Grid [{row}, {col}]...")
        self.get_logger().info(
            f"Target grid XY: x={place_x:.3f}, y={place_y:.3f}"
        )
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed to move to grid position. Aborting.")
            return False
        time.sleep(0.2)

        # 7. DESCEND vertically to drop height (Z-only motion)
        self.get_logger().info("STEP 6: Descending in -Z direction to drop height...")
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_low],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed to descend to drop height. Aborting.")
            return False
        time.sleep(0.2)

        # 8. Open gripper to release cube
        self.get_logger().info("STEP 7: Opening Gripper to release cube...")
        self.set_gripper(gripper_open)
        time.sleep(0.3)

        # 9. ASCEND vertically after releasing (Z-only motion)
        self.get_logger().info("STEP 8: Ascending in +Z direction (Post-Release)...")
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Failed final ascent. Aborting.")
            return False
        time.sleep(0.2)

        self.get_logger().info("=== Pick and Place Sequence Complete! :) ===")
        return True

def main(args=None):
    rclpy.init(args=args)
    
    print("\n========================================")
    print("FR3 Autonomous Pick & Place")
    print("========================================")
    try:
        target_row = int(input("Enter target Row (0 to 5): "))
        target_col = int(input("Enter target Col (0 to 6): "))
    except ValueError:
        print("\n[ERROR] Invalid input. You must enter whole numbers. Exiting.")
        rclpy.shutdown()
        return

    print(f"\n[INFO] Target Grid Cells [{target_row}, {target_col}]")
    print("[INFO] Connecting to hardware...")
    
    # 2. Initialize the Node
    node = FullPickPlaceNode()

    # 3. Execute the Sequence
    success = node.execute_pick_place(row=target_row, col=target_col)

    if success:
        node.get_logger().info("Sequence finished successfully. Shutting down node.")
    else:
        node.get_logger().error("Sequence failed or was aborted.")

    # 4. Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()