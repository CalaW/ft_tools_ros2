#!/usr/bin/env python3
import time
from typing import Callable, NamedTuple

import numpy as np
import rclpy
import rclpy.subscription
import tf2_geometry_msgs  # noqa: F401
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3, Vector3Stamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# TODO parameterize
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
        # TODO parameterize g, frame_id, ft_frame
        g = 9.81
        self.gravity = Vector3Stamped(
            header=Header(frame_id="world", stamp=Time(sec=0, nanosec=0)),
            vector=Vector3(x=0, y=0, z=-g),
        )
        self.ft_frame = "ati_sensing_frame"

    # def init_callback(self) -> None:
    #     self.tf_buffer.can_transform("world", self.ft_frame, 0)

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
        start = time.perf_counter()
        # self.gravity.header.stamp = self.get_clock().now().to_msg()
        g_in_ft = self.tf_buffer.transform(self.gravity, self.ft_frame)
        end = time.perf_counter()
        self.get_logger().info(
            f"tf took {end - start} seconds, {g_in_ft.vector.x}, {g_in_ft.vector.y}, {g_in_ft.vector.z}"
        )
        gravity = np.array([g_in_ft.vector.x, g_in_ft.vector.y, g_in_ft.vector.z])

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
        start = time.perf_counter()

        if self.sample_callback:
            self.sample_callback(SampleResult(ft_mean=mean, gravity=gravity))

        end = time.perf_counter()
        self.get_logger().info(f"gui callback took {end - start} seconds")
        if self.buffer[0].header.frame_id != self.ft_frame:
            self.get_logger().warn(
                f"ft frame mismatch: {self.buffer[0].header.frame_id} != {self.ft_frame}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = FTSamplerNode()
    node.timer = node.create_timer(3, node.trigger_sample)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
