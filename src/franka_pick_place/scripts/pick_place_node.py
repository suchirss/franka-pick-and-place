#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2.moveit2 import MoveIt2
from geometry_msgs.msg import PoseStamped
import time

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
    print("(updated 7th april) Waiting for robot state...")
    time.sleep(2) 

    # 2. Pre-Grasp Pose
    print("Moving to pre-grasp...")
    pose = PoseStamped()
    pose.header.frame_id = "fr3_link0"
    
    # Position: High and center
    pose.pose.position.x = 0.4
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.5 
    
    # Orientation: 180-degree flip around X (pointing down)
    pose.pose.orientation.x = 1.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 0.0
    
    # Execute Pre-Grasp
    moveit2.move_to_pose(pose)
    
    print("Waiting for pre-grasp to complete...")
    pre_grasp_success = moveit2.wait_until_executed()

    # State Machine Logic
    if pre_grasp_success:
        print("[SUCCESS] Pre-grasp reached. Proceeding to descend...")
        
        # 3. Descend slowly
        pose.pose.position.z = 0.2
        moveit2.move_to_pose(pose)
        
        print("Waiting for descent to complete...")
        descend_success = moveit2.wait_until_executed()
        
        if descend_success:
            print("[SUCCESS] Descent complete. Ready for gripper action.")
        else:
            print("[ERROR] Failed during descent execution.")
            
    else:
        print("[ERROR] Aborting: Pre-grasp execution failed!")

    # 4. Graceful Shutdown
    print("Sequence complete. Shutting down in 2 seconds...")
    time.sleep(2.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()