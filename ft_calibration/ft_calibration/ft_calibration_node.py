#!/usr/bin/env python3

import numpy as np
import rclpy
import rclpy.subscription
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

SAMPLE_NUM = 1000


class FTCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("ft_test")
        self.buffer: list[WrenchStamped] = []
        self.timer = self.create_timer(2, self.trigger_sample)
        self.sub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def ft_callback(self, msg: WrenchStamped) -> None:
        self.buffer.append(msg)
        if len(self.buffer) >= SAMPLE_NUM:
            self.destroy_subscription(self.sub)
            self.sub = None
            self.finish_sample()

    def trigger_sample(self) -> None:
        self.get_logger().info("Started sampling")
        self.buffer.clear()
        if self.sub is None:
            self.sub = self.create_subscription(
                WrenchStamped,
                "ft/force_torque_sensor_broadcaster/wrench",
                self.ft_callback,
                qos_profile_sensor_data,
            )

    def finish_sample(self):
        self.get_logger().info("finished sampling")
        mean = np.mean(
            np.array(
                [
                    (
                        msg.wrench.force.x,
                        msg.wrench.force.y,
                        msg.wrench.force.z,
                        msg.wrench.torque.x,
                        msg.wrench.torque.y,
                        msg.wrench.torque.z,
                    )
                    for msg in self.buffer
                ]
            ),
            axis=0,
        )
        self.get_logger().info(f"{mean}")


def main(args=None):
    rclpy.init(args=args)
    node = FTCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
