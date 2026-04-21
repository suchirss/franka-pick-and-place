#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.msg import BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Vector3, Quaternion
import math


class ConstraintsManager:
    """Manages safety constraints for Franka arm operation."""
    
    # Workspace bounds (in meters, relative to fr3_link0 base)
    # Physical setup:
    # - Table: 24" (width X) × 48" (depth Y)
    # - Arm base plate: 8" × 11", bolted to table bottom-left corner
    # - Base plate top-right corner: -15.5" left, -0.5" from table center
    # - Base plate CENTER (robot fr3_link0): -19.5" left, -6" from table center
    # - Safe reach: ±12" forward/backward (X), ±18" left/right (Y), 0 to +18" height (Z) from table center
    # Bounds below are in arm base (fr3_link0) frame, already accounting for offset
    WORKSPACE_BOUNDS = {
        'x_min': 0.05,      # forward-backward minimum (meters)
        'x_max': 0.75,      # forward-backward maximum (meters)
        'y_min': -0.50,     # left-right minimum (meters)
        'y_max': 0.50,      # left-right maximum (meters)
        'z_min': -0.05,     # height minimum (meters, allows slight margin below table)
        'z_max': 0.50,      # height maximum (meters, ~20 inches)
    }
    
    # Velocity/acceleration scaling
    MAX_VELOCITY_SCALE = 0.2        # 20% of maximum speed
    MAX_ACCELERATION_SCALE = 0.2    # 20% of maximum acceleration
    
    # Tolerance for orientation constraint (radians)
    ORIENTATION_TOLERANCE = math.radians(5.0)  # ±5 degrees
    
    @staticmethod
    def create_orientation_constraint(link_name="fr3_hand"):
        """
        Create an orientation constraint to keep gripper pointing downward.
        Gripper down orientation: [1.0, 0.0, 0.0, 0.0] (x, y, z, w in quaternion)
        """
        constraint = OrientationConstraint()
        constraint.link_name = link_name
        constraint.header.frame_id = "fr3_link0"
        
        # Downward orientation (gripper pointing down)
        constraint.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
        
        # Allow small tolerance around the orientation
        constraint.absolute_x_axis_tolerance = ConstraintsManager.ORIENTATION_TOLERANCE
        constraint.absolute_y_axis_tolerance = ConstraintsManager.ORIENTATION_TOLERANCE
        constraint.absolute_z_axis_tolerance = ConstraintsManager.ORIENTATION_TOLERANCE
        constraint.weight = 1.0
        
        return constraint
    
    @staticmethod
    def create_position_constraint(link_name="fr3_hand"):
        """
        Create a position constraint to keep end-effector within safe workspace bounds.
        """
        constraint = PositionConstraint()
        constraint.link_name = link_name
        constraint.header.frame_id = "fr3_link0"
        
        # Define bounding box (conservative workspace)
        bounding_volume = BoundingVolume()
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        
        # Box dimensions (width, depth, height)
        bounds = ConstraintsManager.WORKSPACE_BOUNDS
        box.dimensions = [
            bounds['x_max'] - bounds['x_min'],      # x-extent (forward-backward)
            bounds['y_max'] - bounds['y_min'],      # y-extent (left-right)
            bounds['z_max'] - bounds['z_min'],      # z-extent (vertical)
        ]
        
        # Box center position
        box_center = Pose()
        box_center.position.x = (bounds['x_max'] + bounds['x_min']) / 2.0
        box_center.position.y = (bounds['y_max'] + bounds['y_min']) / 2.0
        box_center.position.z = (bounds['z_max'] + bounds['z_min']) / 2.0
        box_center.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        bounding_volume.primitives.append(box)
        bounding_volume.primitive_poses.append(box_center)
        
        constraint.constraint_region = bounding_volume
        constraint.weight = 1.0
        
        return constraint
    
    @staticmethod
    def create_safe_constraints():
        """
        Create a Constraints message with both orientation and position constraints.
        """
        constraints = Constraints()
        constraints.orientation_constraints.append(
            ConstraintsManager.create_orientation_constraint()
        )
        constraints.position_constraints.append(
            ConstraintsManager.create_position_constraint()
        )
        return constraints


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