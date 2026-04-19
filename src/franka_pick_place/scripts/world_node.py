#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class WorldNode(Node):
    def __init__(self):
        super().__init__('world_node')
        self.publisher = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.timer = self.create_timer(1.0, self.spawn_world)

    def spawn_world(self):
        table = CollisionObject()
        table.header.frame_id = 'fr3_link0'
        table.id = 'lab_table'

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.61, 1.22, 0.04]

        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.0
        pose.position.z = -0.04

        table.primitives.append(box)
        table.primitive_poses.append(pose)
        table.operation = CollisionObject.ADD

        cube = CollisionObject()
        cube.header.frame_id = 'fr3_link0'
        cube.id = 'test_cube'

        c_box = SolidPrimitive()
        c_box.type = SolidPrimitive.BOX
        c_box.dimensions = [0.05, 0.05, 0.05]

        c_pose = Pose()
        c_pose.position.x = 0.5
        c_pose.position.y = 0.0
        c_pose.position.z = 0.025

        cube.primitives.append(c_box)
        cube.primitive_poses.append(c_pose)
        cube.operation = CollisionObject.ADD

        self.publisher.publish(table)
        self.publisher.publish(cube)
        self.get_logger().info('World objects spawned.')
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = WorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()