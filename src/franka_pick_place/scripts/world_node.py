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
        # Timer runs once immediately
        self.timer = self.create_timer(1.0, self.spawn_world)

    def spawn_world(self):
        # 1. Define Lab Table
        table = CollisionObject()
        table.header.frame_id = 'fr3_link0'
        table.id = 'lab_table'
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.61, 1.22, 0.04]
        
        pose = Pose()
        pose.position.x = 0.5 # Change as needed
        pose.position.z = -0.04 # Change as needed
        
        table.primitives.append(box)
        table.primitive_poses.append(pose)
        table.operation = CollisionObject.ADD

        # 2. Define Test Cube
        cube = CollisionObject()
        cube.header.frame_id = 'fr3_link0'
        cube.id = 'test_cube'
        
        c_box = SolidPrimitive()
        c_box.type = SolidPrimitive.BOX
        c_box.dimensions = [0.05, 0.05, 0.05]
        
        c_pose = Pose()
        c_pose.position.x = 0.5
        c_pose.position.y = 0.0
        c_pose.position.z = 0.005
        
        cube.primitives.append(c_box)
        cube.primitive_poses.append(c_pose)
        cube.operation = CollisionObject.ADD

        # 3. Publish and then DESTROY timer to stop loop
        self.publisher.publish(table)
        self.publisher.publish(cube)
        self.get_logger().info('World objects spawned. Shutting down timer...')
        self.timer.cancel() 

def main():
    rclpy.init()
    node = WorldNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
