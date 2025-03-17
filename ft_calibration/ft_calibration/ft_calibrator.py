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
        # Use a list to store sample data, each element is (gravity, ft_raw)
        self.measurements = []

    def add_measurement(self, gravity: np.ndarray, ft_raw: np.ndarray) -> None:
        """
        Add measurement data

        Parameters:
            gravity: np.array, contains 3 elements [gx, gy, gz]
            ft_raw:  np.array, contains 6 elements [fx, fy, fz, tx, ty, tz]
        """
        gravity = np.asarray(gravity).flatten()
        ft_raw = np.asarray(ft_raw).flatten()
        if gravity.shape != (3,) or ft_raw.shape != (6,):
            raise ValueError("gravity must have 3 elements, ft_raw must have 6 elements")
        self.measurements.append((gravity, ft_raw))

    def delete_measurement(self, index: int) -> None:
        """
        Delete the specified measurement data

        Parameters:
            index: Index of the data to be deleted (starting from 0)
        """
        if index < 0 or index >= len(self.measurements):
            raise IndexError("Measurement data index out of range")
        del self.measurements[index]

    def get_calibration(self) -> CalibrationResult:
        """
        Get calibration result

        Returns:
            CalibrationResult object, including mass, center of gravity, force bias, and torque bias
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
        Perform calibration calculation on all measurement data using least squares method

        Returns:
            np.array containing 10 calibration parameters
        """
        if not self.measurements:
            raise ValueError("No measurement data available for calibration")

        H_list = []
        Z_list = []
        for gravity, ft_raw in self.measurements:
            H_list.append(self._get_measurement_matrix(gravity))
            Z_list.append(ft_raw.flatten())
        H = np.vstack(H_list)
        Z = np.concatenate(Z_list)
        # Solve the least squares solution of H * params = Z
        calib_params, _, _, _ = np.linalg.lstsq(H, Z, rcond=None)
        return calib_params

    def _get_measurement_matrix(self, gravity: np.ndarray) -> np.ndarray:
        """
        Construct measurement matrix H based on the given gravity vector

        Parameters:
            gravity: np.array, contains 3 elements [gx, gy, gz]

        Returns:
            H: 6x10 np.array
        """
        g = np.asarray(gravity).flatten()
        # w, alpha, a are all zero vectors
        w = np.zeros(3)
        alpha = np.zeros(3)
        a = np.zeros(3)

        H = np.zeros((6, 10))
        # first three rows, columns 4 to 9 filled with identity matrix
        H[:3, 4:10] = np.eye(6)[:3]
        # last three rows, columns 4 to 9 filled with identity matrix
        H[3:6, 4:10] = np.eye(6)[3:6]
        # dynamic parameters of the first three rows: a - g (a is a zero vector, so it is -g)
        H[:3, 0] = -g

        # rotational coupling terms of the first three rows
        # (since w and alpha are zero, mainly keep the structure consistent)
        H[0, 1] = -(w[1] ** 2) - w[2] ** 2
        H[0, 2] = w[0] * w[1] - alpha[2]
        H[0, 3] = w[0] * w[2] + alpha[1]

        H[1, 1] = w[0] * w[1] + alpha[2]
        H[1, 2] = -(w[0] ** 2) - w[2] ** 2
        H[1, 3] = w[1] * w[2] - alpha[0]

        H[2, 1] = w[0] * w[2] - alpha[1]
        H[2, 2] = w[1] * w[2] + alpha[0]
        H[2, 3] = -(w[1] ** 2) - w[0] ** 2

        # cross-coupling terms of the last three rows
        H[3, 2] = a[2] - g[2]
        H[3, 3] = -a[1] + g[1]

        H[4, 1] = -a[2] + g[2]
        H[4, 3] = a[0] - g[0]

        H[5, 1] = a[1] - g[1]
        H[5, 2] = -a[0] + g[0]

        return H
