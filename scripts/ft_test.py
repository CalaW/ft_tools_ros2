#!/usr/bin/env python3

import numpy as np
import rclpy
import rclpy.subscription
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

SAMPLE_NUM = 1000


class FtTest(Node):
    def __init__(self) -> None:
        super().__init__("ft_test")
        self.buffer: list[WrenchStamped] = []
        self.timer = self.create_timer(2, self.trigger_sample)
        self.sub = None

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
    ft_test = FtTest()
    rclpy.spin(ft_test)
    ft_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
