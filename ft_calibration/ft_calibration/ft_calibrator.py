from pathlib import Path
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
        # 初始化测量次数、H矩阵和Z向量
        self.num_meas = 0
        self.H = np.empty((0, 10))  # H矩阵初始为空，每次添加一个 6x10 块
        self.Z = np.empty(0)  # Z向量初始为空，每次添加一个 6 维向量

    def add_measurement(self, gravity: np.ndarray, ft_raw: np.ndarray) -> None:
        """
        添加测量数据

        参数:
            gravity: np.array，包含3个元素，顺序为 [gx, gy, gz]
            ft_raw:  np.array，包含6个元素，顺序为 [fx, fy, fz, tx, ty, tz]

        注意: 假定用户已经完成坐标系的转换工作。
        """
        self.num_meas += 1
        h = self._get_measurement_matrix(gravity)
        # 确保 ft_raw 为1维数组（6个元素）
        z = ft_raw.flatten()
        # 动态拼接H矩阵和Z向量
        if self.H.size:
            self.H = np.vstack([self.H, h])
            self.Z = np.concatenate([self.Z, z])
        else:
            self.H = h
            self.Z = z

    def get_calibration(self) -> CalibrationResult:
        res = self.get_calibration_raw()
        return CalibrationResult(
            mass=res[0],
            cog=Vector3(x=res[1], y=res[2], z=res[3]),
            f_bias=Vector3(res[4], res[5], res[6]),
            t_bias=Vector3(res[7], res[8], res[9]),
        )

    def get_calibration_raw(self) -> np.ndarray:
        """
        利用最小二乘法求解校准参数

        返回:
            一个包含10个校准参数的 np.array
        """
        if self.num_meas == 0:
            raise ValueError("没有可用的测量数据进行校准")
        # 利用 numpy.linalg.lstsq 求解 H * params = Z
        calib_params, residuals, rank, s = np.linalg.lstsq(self.H, self.Z, rcond=None)
        return calib_params

    def _get_measurement_matrix(self, gravity: np.ndarray) -> np.ndarray:
        """
        根据给定的重力向量构造测量矩阵 H

        参数:
            gravity: np.array，包含3个元素，顺序为 [gx, gy, gz]

        返回:
            H: 一个 6x10 的 np.array

        说明:
            为了与原始代码保持一致，角速度 w、角加速度 alpha 以及加速度 a 均设置为零向量。
        """
        # 将输入转换为1维数组
        g = gravity.flatten()  # [gx, gy, gz]
        # 将 w, alpha, a 均设为零向量
        w = np.zeros(3)
        alpha = np.zeros(3)
        a = np.zeros(3)

        # 初始化6x10的零矩阵
        H = np.zeros((6, 10))  # noqa: N806

        # # --- 第一部分: 对于行 0~2，填充列 4~9 (单位子矩阵)
        H[0:3, 4:10] = np.eye(6)[0:3, 0:6]  # 设置单位矩阵部分

        # --- 第二部分: 对于行 3~5的初始赋值
        for i in range(3, 6):
            H[i, 0] = 0.0
        H[3, 1] = 0.0
        H[4, 2] = 0.0
        H[5, 3] = 0.0

        # --- 第三部分: 前三行的动态参数赋值：H(i,0) = a(i) - g(i) （a为零向量，所以为 -g）
        for i in range(3):
            H[i, 0] = a[i] - g[i]

        # --- 第四部分: 前三行的旋转耦合项（w 和 alpha 均为零，因此这些项均为 0）
        H[0, 1] = -w[1] * w[1] - w[2] * w[2]
        H[0, 2] = w[0] * w[1] - alpha[2]
        H[0, 3] = w[0] * w[2] + alpha[1]

        H[1, 1] = w[0] * w[1] + alpha[2]
        H[1, 2] = -w[0] * w[0] - w[2] * w[2]
        H[1, 3] = w[1] * w[2] - alpha[0]

        H[2, 1] = w[0] * w[2] - alpha[1]
        H[2, 2] = w[1] * w[2] + alpha[0]
        H[2, 3] = -w[1] * w[1] - w[0] * w[0]

        # --- 第五部分: 后三行的交叉耦合项
        H[3, 2] = a[2] - g[2]  # 实际值为 -g[2]
        H[3, 3] = -a[1] + g[1]  # 实际值为 g[1]

        H[4, 1] = -a[2] + g[2]  # 实际值为 g[2]
        H[4, 3] = a[0] - g[0]  # 实际值为 -g[0]

        H[5, 1] = a[1] - g[1]  # 实际值为 -g[1]
        H[5, 2] = -a[0] + g[0]  # 实际值为 g[0]

        # --- 第六部分: 对于行 3~5，填充列 4~9，用单位矩阵覆盖
        H[3:6, 4:10] = np.eye(6)[3:6, 0:6]

        return H


# --------------------- 用例示例 ---------------------
if __name__ == "__main__":
    # 初始化校准器
    calib = FTCalibrator()

    # each line is gx, gy, gz, fx, fy, fz, tx, ty, tz
    file = Path(__file__).parent / "example_data.txt"
    with file.open("r") as f:
        for line in f:
            data = [float(x) for x in line.strip().split(" ")]
            gravity = np.array(data[:3])
            ft_raw = np.array(data[3:])
            calib.add_measurement(gravity, ft_raw)

    # 执行最小二乘校准
    params = calib.get_calibration()
