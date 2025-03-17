from typing import NamedTuple

import numpy as np


class Vector3(NamedTuple):
    x: float
    y: float
    z: float


class CalibrationResult(NamedTuple):
    mass: float
    cog: Vector3
    f_bias: Vector3
    t_bias: Vector3


class FTCalibrator:
    def __init__(self):
        # 使用列表存储采样数据，每个元素为 (gravity, ft_raw)
        self.measurements = []

    def add_measurement(self, gravity: np.ndarray, ft_raw: np.ndarray) -> None:
        """
        添加测量数据

        参数:
            gravity: np.array，包含3个元素 [gx, gy, gz]
            ft_raw:  np.array，包含6个元素 [fx, fy, fz, tx, ty, tz]
        """
        gravity = np.asarray(gravity).flatten()
        ft_raw = np.asarray(ft_raw).flatten()
        if gravity.shape != (3,) or ft_raw.shape != (6,):
            raise ValueError("gravity 必须有3个元素，ft_raw 必须有6个元素")
        self.measurements.append((gravity, ft_raw))

    def delete_measurement(self, index: int) -> None:
        """
        删除指定序号的测量数据

        参数:
            index: 待删除数据的索引（从0开始）
        """
        if index < 0 or index >= len(self.measurements):
            raise IndexError("测量数据索引超出范围")
        del self.measurements[index]

    def get_calibration(self) -> CalibrationResult:
        """
        获取校准结果

        返回:
            CalibrationResult 对象，包括质量、重心、力偏差和力矩偏差
        """
        res = self.get_calibration_raw()
        return CalibrationResult(
            mass=res[0],
            cog=Vector3(x=res[1], y=res[2], z=res[3]),
            f_bias=Vector3(x=res[4], y=res[5], z=res[6]),
            t_bias=Vector3(x=res[7], y=res[8], z=res[9]),
        )

    def get_calibration_raw(self):
        """
        利用最小二乘法对所有测量数据进行校准计算

        返回:
            包含10个校准参数的 np.array
        """
        if not self.measurements:
            raise ValueError("没有可用的测量数据进行校准")

        H_list = []
        Z_list = []
        for gravity, ft_raw in self.measurements:
            H_list.append(self._get_measurement_matrix(gravity))
            Z_list.append(ft_raw.flatten())
        H = np.vstack(H_list)
        Z = np.concatenate(Z_list)
        # 求解 H * params = Z 的最小二乘解
        calib_params, _, _, _ = np.linalg.lstsq(H, Z, rcond=None)
        return calib_params

    def _get_measurement_matrix(self, gravity: np.ndarray) -> np.ndarray:
        """
        根据给定的重力向量构造测量矩阵 H

        参数:
            gravity: np.array，包含3个元素 [gx, gy, gz]

        返回:
            H: 6x10 的 np.array
        """
        g = np.asarray(gravity).flatten()
        # w, alpha, a 均设为零向量
        w = np.zeros(3)
        alpha = np.zeros(3)
        a = np.zeros(3)

        H = np.zeros((6, 10))
        # 第一部分: 前三行，列4到9填充单位矩阵
        H[:3, 4:10] = np.eye(6)[:3]
        # 第六部分: 后三行，列4到9填充单位矩阵
        H[3:6, 4:10] = np.eye(6)[3:6]
        # 第三部分: 前三行的动态参数赋值: a - g (a为零向量，所以为 -g)
        H[:3, 0] = -g

        # 第四部分: 前三行的旋转耦合项 (由于 w 和 alpha 为零，这里主要保持结构一致)
        H[0, 1] = -(w[1] ** 2) - w[2] ** 2
        H[0, 2] = w[0] * w[1] - alpha[2]
        H[0, 3] = w[0] * w[2] + alpha[1]

        H[1, 1] = w[0] * w[1] + alpha[2]
        H[1, 2] = -(w[0] ** 2) - w[2] ** 2
        H[1, 3] = w[1] * w[2] - alpha[0]

        H[2, 1] = w[0] * w[2] - alpha[1]
        H[2, 2] = w[1] * w[2] + alpha[0]
        H[2, 3] = -(w[1] ** 2) - w[0] ** 2

        # 第五部分: 后三行的交叉耦合项
        H[3, 2] = a[2] - g[2]
        H[3, 3] = -a[1] + g[1]

        H[4, 1] = -a[2] + g[2]
        H[4, 3] = a[0] - g[0]

        H[5, 1] = a[1] - g[1]
        H[5, 2] = -a[0] + g[0]

        return H
