#!/usr/bin/env python3
import copy
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand


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
        
        # 2. Setup Action Client for the Gripper
        # Verify this topic name with your franka_ros2 launch files (usually /fr3_gripper/gripper_action)
        self.gripper_client = ActionClient(self, GripperCommand, '/fr3_gripper/gripper_action')
        
        self.target_received = False
        self.robot_target_x = None
        self.robot_target_y = None

        # 3. Setup the Vision Subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            '/vision/cube_pose',  
            self.vision_callback,
            10
        )
        self.get_logger().info("Ready. Waiting for camera vision data...")

    def set_gripper(self, width, max_effort=20.0):
        """Helper function to open/close the Franka Hand"""
        self.get_logger().info(f"Moving gripper to {width}m...")
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available!")
            return False
            
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width # 0.08 is fully open, 0.0 is closed
        goal_msg.command.max_effort = max_effort
        
        # Send goal and wait (synchronous execution for the state machine)
        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        time.sleep(0.5) # Pause to settle fingers
        return True

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
        
        #TODO - verify
        # --- CONSTANTS ---
        safe_z_high = 0.50       # Hover height (50cm) - safety height
        safe_z_low = 0.02        # Height at which grippers close (2cm) - can increase to 0.10 for first run to avoid hitting table
        gripper_open = 0.08      # 8cm wide
        gripper_closed = 0.0381  # 1.5 inches (size of cube) - will close at either this or max effort - see helper action above
        downward_orientation = [1.0, 0.0, 0.0, 0.0] # Default Gripper straight down orientation (x, y, z, w)

        # Ensure gripper is open before starting
        self.set_gripper(gripper_open)
        
        # PICK UP PHASE=================================

        # 1. Setup Pre-Grasp Pose
        self.get_logger().info("STEP 1: Moving to Pre-Grasp...")
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        ) # if not centered above cube, press e-stop here!
        if not self.moveit2.wait_until_executed(): 
            self.get_logger().error("Failed to reach Pre-Grasp. Aborting sequence.")
            return False
        
        # 2. Descend to target
        self.get_logger().info("STEP 2: Descending to Grasp Z-Height...")
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_low],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed(): return False

        # 3. Close gripper and grab cube
        self.get_logger().info("STEP 3: Closing Gripper...")
        self.set_gripper(gripper_closed)

        #4. Lift up after closing gripper
        self.get_logger().info("STEP 4: Ascending (Post-Grasp)...")
        self.moveit2.move_to_pose(
            position=[self.robot_target_x, self.robot_target_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        ) 
        if not self.moveit2.wait_until_executed(): return False

        # PLACE PHASE=================================

        # 5. Calculate Target Grid Coordinates
        cell_size_m = 0.0381 #TODO Match with irl dims
        board_origin_x = 0.45  # Same as board_offset_x
        board_origin_y = -0.15 # Same as board_offset_y
        place_x = board_origin_x + (row * cell_size_m) # x = 0.45+2*0.05 = 0.55
        place_y = board_origin_y + (col * cell_size_m)  # y = -0.15+3*0.05 = 0

        # 6. Motion to place above gridspace ****(row, col)****
        self.get_logger().info(f"STEP 5: Translating to Grid [{row}, {col}]...")
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed(): return False

        # 7. Descend to the table safely
        self.get_logger().info("STEP 6: Descending to Drop Z-Height...")
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_low],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed(): return False

        # 8. Open gripper to release
        self.get_logger().info("STEP 7: Opening Gripper (Releasing)...")
        self.set_gripper(gripper_open)

        # 9. Lift up after releasing
        self.get_logger().info("STEP 8: Ascending (Post-Place)...")
        self.moveit2.move_to_pose(
            position=[place_x, place_y, safe_z_high],
            quat_xyzw=downward_orientation,
            frame_id="fr3_link0"
        )
        if not self.moveit2.wait_until_executed(): return False

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