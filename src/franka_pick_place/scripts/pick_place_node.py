#!/usr/bin/env python3
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
            return # Only execute once per run
            
        self.target_received = True
        self.get_logger().info("Target acquired! Initiating Pick Sequence.")

        # -- PHASE 0: Open Gripper --
        self.set_gripper(0.08) # Open to 8cm

        safe_z_height = 0.005 # Center of cube on table
        hover_z_height = 0.20 # 20cm above table
        
        # -- PHASE 1: Pre-Grasp --
        target_pose = PoseStamped()
        target_pose.header.frame_id = "fr3_link0"
        target_pose.pose.position.x = msg.pose.position.x
        target_pose.pose.position.y = msg.pose.position.y
        target_pose.pose.position.z = hover_z_height 
        
        # Point straight down (180 deg flip around X)
        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.w = 0.0

        self.get_logger().info("Moving to Pre-Grasp...")
        self.moveit2.move_to_pose(target_pose)
        if not self.moveit2.wait_until_executed():
            self.get_logger().error("Pre-grasp failed.")
            return

        # -- PHASE 2: Descend --
        self.get_logger().info("Descending...")
        target_pose.pose.position.z = safe_z_height 
        self.moveit2.move_to_pose(target_pose)
        self.moveit2.wait_until_executed()
        
        # -- PHASE 3: Grasp --
        self.get_logger().info("Grasping object...")
        self.set_gripper(0.048, max_effort=30.0) 
        
        # NOTE FIX: Attach the object to the robot's hand
        self.get_logger().info("Attaching cube to the planning scene...")
        
        # You need the exact ID you gave the cube in your world_node.py
        cube_id = "target_cube" 
        
        # Attach the cube to the hand link so MoveIt knows they are one object
        # (Check pymoveit2 documentation for exact syntax, usually it's tied to the PlanningScene)
        # Pseudocode for the logic:
        # planning_scene.attach_collision_object(object_id=cube_id, link_name="fr3_hand")

        # -- PHASE 4: Retreat --
        self.get_logger().info("Lifting...")
        target_pose.pose.position.z = hover_z_height 
        self.moveit2.move_to_pose(target_pose)
        self.moveit2.wait_until_executed()

        self.get_logger().info("Pick complete! Ready for drop-off trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = FullPickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()