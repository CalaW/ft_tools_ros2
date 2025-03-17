#!/usr/bin/env python3

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401
from builtin_interfaces.msg import Time
from ft_compensation import FTCompensator
from geometry_msgs.msg import Vector3, Vector3Stamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class FTCompensationNode(Node):
    def __init__(self):
        super().__init__("ft_compensation")

        calib_params = np.array(
            [
                0.47623183521135465,
                -0.00041899723931493327,
                -0.0005572845786021574,
                0.04241625174721376,
                -3.632659850281348,
                -9.21982189209215,
                2.9150709092032887,
                -0.19825460420788116,
                -0.045504989075480506,
                -0.2513034824363129,
            ]
        )
        self.compensator = FTCompensator(calib_params)
        self.subscription = self.create_subscription(
            WrenchStamped,
            "ft/force_torque_sensor_broadcaster/wrench",
            self.callback,
            qos_profile_sensor_data,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # TODO parameterize g, frame_id, ft_frame
        g = 9.81
        self.gravity = Vector3Stamped(
            header=Header(frame_id="world", stamp=Time(sec=0, nanosec=0)),
            vector=Vector3(x=0, y=0, z=-g),
        )
        self.ft_frame = "ati_sensing_frame"

    def callback(self, msg: WrenchStamped):
        raw_ft = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )
        try:
            g_in_ft = self.tf_buffer.transform(self.gravity, self.ft_frame)
        except TransformException as ex:
            self.get_logger().warn(f"Failed to transform gravity: {ex}", throttle_duration_sec=1)
            return
        gravity = np.array([g_in_ft.vector.x, g_in_ft.vector.y, g_in_ft.vector.z])
        compensated_force, compensated_torque = self.compensator.compensate(raw_ft, gravity)
        msg.wrench.force.x = compensated_force[0]
        msg.wrench.force.y = compensated_force[1]
        msg.wrench.force.z = compensated_force[2]
        msg.wrench.torque.x = compensated_torque[0]
        msg.wrench.torque.y = compensated_torque[1]
        msg.wrench.torque.z = compensated_torque[2]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FTCompensationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
