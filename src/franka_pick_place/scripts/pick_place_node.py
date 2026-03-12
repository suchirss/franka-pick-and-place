#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2.moveit2 import MoveIt2
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration

def main():
    rclpy.init()
    node = rclpy.create_node('pick_place_node')
    
    fr3_joints = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]

    moveit2 = MoveIt2(
        node, 
        group_name="fr3_arm", 
        joint_names=fr3_joints,
        base_link_name="fr3_link0", 
        end_effector_name="fr3_hand"
    )

    # 1. Wait for Joint States to stabilize
    print("Waiting for robot state...")
    import time
    time.sleep(2) 

    # 2. Pre-Grasp Pose
    print("Moving to pre-grasp...")
    pose = PoseStamped()
    pose.header.frame_id = "fr3_link0"
    
    # Position: High and center
    pose.pose.position.x = 0.4
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.5 # Much higher to avoid the table initially
    
    # Orientation: This is a 180-degree flip around X (pointing down)
    # Using a slightly different setup to help the solver
    pose.pose.orientation.x = 1.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 0.0
    
    # INCREASE PLANNING TIME (Temporary override)
    # moveit2.set_planning_time(10.0) 
    success = moveit2.move_to_pose(pose)

    if not success:
        print("Pre-grasp planning failed!")

    # 3. Descend slowly (Small increments help IK solvers)
    print("Descending...")
    pose.pose.position.z = 0.2
    moveit2.move_to_pose(pose)

    print("Sequence complete.")
    rclpy.shutdown()

if __name__ == '__main__':
    main()