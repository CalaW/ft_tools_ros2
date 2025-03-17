#!/usr/bin/env python3

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401
from builtin_interfaces.msg import Time
from ft_compensation import FTCompensator
from geometry_msgs.msg import Vector3, Vector3Stamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class FTCompensationNode(Node):
    def __init__(self):
        super().__init__("ft_compensation")

        calib_params = np.array(
            [
                -0.6808681364695239,
                -0.0007039575712726176,
                0.0010365292746074284,
                -0.0757526200201236,
                -0.03745466131196212,
                -0.04386771661018282,
                -6.752788407036707,
                0.00734601382527324,
                -0.0015608840790269486,
                0.006534742397436891,
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
        self.publisher = self.create_publisher(
            WrenchStamped, "/wrench_compensated", qos_profile_system_default
        )

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
            self.get_logger().warn(
                f"Failed to transform gravity: {ex}", throttle_duration_sec=1, skip_first=True
            )
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
