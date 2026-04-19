#!/usr/bin/env python3

import copy
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand
from pymoveit2.moveit2 import MoveIt2


class FullPickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        fr3_joints = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

        self.moveit2 = MoveIt2(
            node=self,
            group_name="fr3_arm",
            joint_names=fr3_joints,
            base_link_name="fr3_link0",
            end_effector_name="fr3_hand",
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/fr3_gripper/gripper_action'
        )

        self.target_received = False

        self.subscription = self.create_subscription(
            PoseStamped,
            '/vision/cube_pose',
            self.vision_callback,
            10
        )

        self.get_logger().info("Ready. Waiting for camera vision data...")

    def set_gripper(self, width, max_effort=20.0):
        self.get_logger().info(f"Moving gripper to width={width:.3f} m")

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available")
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width
        goal_msg.command.max_effort = max_effort

        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Gripper action complete")
        return True

    def vision_callback(self, msg):
        if self.target_received:
            return

        self.target_received = True
        self.get_logger().info("Target acquired from camera. Starting sequence.")

        board_offset_x = 0.3
        board_offset_y = 0.1778

        robot_target_x = msg.pose.position.x + board_offset_x
        robot_target_y = msg.pose.position.y + board_offset_y

        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header.frame_id = "fr3_link0"
        pre_grasp_pose.header.stamp = self.get_clock().now().to_msg()

        pre_grasp_pose.pose.position.x = robot_target_x
        pre_grasp_pose.pose.position.y = robot_target_y
        pre_grasp_pose.pose.position.z = 0.50

        pre_grasp_pose.pose.orientation.x = 1.0
        pre_grasp_pose.pose.orientation.y = 0.0
        pre_grasp_pose.pose.orientation.z = 0.0
        pre_grasp_pose.pose.orientation.w = 0.0

        self.get_logger().info(
            f"Moving to pre-grasp: x={robot_target_x:.3f}, y={robot_target_y:.3f}"
        )

        self.moveit2.move_to_pose(pre_grasp_pose)
        success = self.moveit2.wait_until_executed()

        if not success:
            self.get_logger().error("Pre-grasp failed. Aborting.")
            return

        self.get_logger().info("Pre-grasp reached. Descending.")

        descend_pose = copy.deepcopy(pre_grasp_pose)
        descend_pose.header.stamp = self.get_clock().now().to_msg()
        descend_pose.pose.position.z = 0.20

        self.moveit2.move_to_pose(descend_pose)
        success = self.moveit2.wait_until_executed()

        if not success:
            self.get_logger().error("Descend failed. Aborting.")
            return

        self.get_logger().info("Ready to close gripper.")
        self.set_gripper(0.02, max_effort=20.0)


def main(args=None):
    rclpy.init(args=args)
    node = FullPickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()