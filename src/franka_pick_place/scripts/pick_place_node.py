#!/usr/bin/env python3
import copy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pymoveit2.moveit2 import MoveIt2
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
        return True

    def vision_callback(self, msg):
        
        if self.target_received:
            return
        
        self.target_received = True
        self.get_logger().info("Target acquired from camera! Initiating sequence.")

        # --- THE TAPE-DOWN OFFSETS (IN METERS) ---
        #TODO
        # UPDATE !! 
        board_offset_x = 0.45   # Distance forward to Marker 15
        board_offset_y = -0.15  # Distance left/right to Marker 15 (Negative = Right)
        
        # 1. Translate Camera Data (Relative to Board) -> Robot Data (Relative to Base)
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
        
        # 2. Setup Pre-Grasp Pose (High Z for safety)
        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header.frame_id = "fr3_link0"
        
        pre_grasp_pose.pose.position.x = self.robot_target_x
        pre_grasp_pose.pose.position.y = self.robot_target_y
        pre_grasp_pose.pose.position.z = 0.50 # 50cm above the table
        
        # Gripper straight down
        pre_grasp_pose.pose.orientation.x = 1.0
        pre_grasp_pose.pose.orientation.y = 0.0
        pre_grasp_pose.pose.orientation.z = 0.0
        pre_grasp_pose.pose.orientation.w = 0.0

        # 3. Execute Pre-Grasp (if pre-grasp successful, can implement other steps)
        self.get_logger().info(f"Moving to Pre-Grasp: X={self.robot_target_x:.3f}, Y={self.robot_target_y:.3f}")
        self.moveit2.move_to_pose(pre_grasp_pose)
        success = self.moveit2.wait_until_executed()

        if success:
            self.get_logger().info("[SUCCESS] Pre-grasp reached. Descending...")
            
            # 4. Descend to the table safely
            safe_z_height = 0.20 
            # 2cm above the table/base 
            #TODO # UPDATE HEIGHT TO 2CM AFTER 20CM SAFE TEST
            
            descend_pose = copy.deepcopy(pre_grasp_pose)
            descend_pose.pose.position.z = safe_z_height 
            
            self.moveit2.move_to_pose(descend_pose)
            self.moveit2.wait_until_executed()
            
            self.get_logger().info("Ready to close gripper!")

        else:
            self.get_logger().error("Pre-grasp failed. Aborting.")
            return False

        #TODO Add close gripper command

        #TODO Add lift up after closing gripper

        #TODO Add motion to place above gridspace ****(row, col)****

        #TODO Add descend to the table safely

        #TODO Add open gripper to release

        #TODO Add lift up after releasing

        # return true if all steps successful, false if any step failed
        return True

def main(args=None):
    rclpy.init(args=args)
    node = FullPickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()