#!/usr/bin/env python3

from typing import Callable, NamedTuple

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


class SampleResult(NamedTuple):
    ft_mean: np.ndarray
    gravity: np.ndarray  # in Newton, length should be g


class FTSamplerNode(Node):
    def __init__(self) -> None:
        super().__init__("ft_test")
        self.buffer: list[WrenchStamped] = []
        self.sub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sample_callback: Callable | None = None

    def set_sample_callback(self, sample_callback: Callable):
        self.sample_callback = sample_callback

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

    def finish_sample(self) -> None:
        self.get_logger().info("finished sampling")
        samples = np.array(
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
        )
        mean = np.mean(samples, axis=0)
        if self.sample_callback:
            self.sample_callback(SampleResult(ft_mean=mean, gravity=np.array([0, 1, 0])))


def main(args=None):
    rclpy.init(args=args)
    node = FTSamplerNode()
    node.timer = node.create_timer(3, node.trigger_sample)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
