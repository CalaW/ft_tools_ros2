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


measurement_type = list[tuple[np.ndarray, np.ndarray]]


class FTCalibrator:
    def __init__(self) -> None:
        # Use a list to store sample data, each element is (gravity, ft_raw)
        self._measurements: measurement_type = []

    @property
    def measurements(self) -> measurement_type:
        return self._measurements

    def add_measurement(self, gravity: np.ndarray, ft_raw: np.ndarray) -> None:
        """
        Add measurement data

        Parameters:
            gravity: np.array, contains 3 elements [gx, gy, gz], in sensor frame
            ft_raw:  np.array, contains 6 elements [fx, fy, fz, tx, ty, tz], in sensor frame
        """
        gravity = np.asarray(gravity).flatten()
        ft_raw = np.asarray(ft_raw).flatten()
        if gravity.shape != (3,) or ft_raw.shape != (6,):
            raise ValueError("gravity must have 3 elements, ft_raw must have 6 elements")
        self._measurements.append((gravity, ft_raw))

    def delete_measurement(self, index: int) -> None:
        """
        Delete the specified measurement data

        Parameters:
            index: Index of the data to be deleted (starting from 0)
        """
        if index < 0 or index >= len(self._measurements):
            raise IndexError("Measurement data index out of range")
        del self._measurements[index]

    def get_calibration(self) -> CalibrationResult:
        res = np.asarray(self.get_calibration_raw(), dtype=float)

        m = float(res[0])
        if abs(m) < 1e-12:
            raise ValueError("Estimated mass is ~0; cannot compute CoG.")

        p = res[1:4]          # p = m*c
        c = p / m             # c = p/m

        return CalibrationResult(
            mass=m,
            cog=Vector3(x=float(c[0]), y=float(c[1]), z=float(c[2])),
            f_bias=Vector3(x=float(res[4]), y=float(res[5]), z=float(res[6])),
            t_bias=Vector3(x=float(res[7]), y=float(res[8]), z=float(res[9])),
        )

    def get_calibration_raw(self) -> np.ndarray:
        """
        Perform calibration calculation on all measurement data using least squares method

        Returns:
            np.array containing 10 calibration parameters
        """
        if not self._measurements:
            raise ValueError("No measurement data available for calibration")

        H_list = []
        Z_list = []
        for gravity, ft_raw in self._measurements:
            H_list.append(self._get_measurement_matrix(gravity))
            Z_list.append(ft_raw)
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
        g = np.asarray(gravity, dtype=float).reshape(3)

        def skew(v: np.ndarray) -> np.ndarray:
            x, y, z = v
            return np.array(
                [
                    [0.0, -z,  y],
                    [z,  0.0, -x],
                    [-y, x,  0.0],
                ],
                dtype=float,
            )

        H = np.zeros((6, 10), dtype=float)

        # Force: F = m*g + f_bias
        H[0:3, 0] = g
        H[0:3, 4:7] = np.eye(3)

        # Torque: tau = (p x g) + t_bias, where p = m*c
        # p x g = -skew(g) @ p
        H[3:6, 1:4] = -skew(g)
        H[3:6, 7:10] = np.eye(3)

        return H

    def save_calibration(self, filename: str) -> None:
        """
        Save the calibration result to a file

        Parameters:
            filename: Path to save the calibration result
        """
        calib_params = self.get_calibration()
        data_str = (
            f"mass: {calib_params.mass}\n"
            f"cog:\n"
            f"  - {calib_params.cog.x}\n"
            f"  - {calib_params.cog.y}\n"
            f"  - {calib_params.cog.z}\n"
            f"f_bias:\n"
            f"  - {calib_params.f_bias.x}\n"
            f"  - {calib_params.f_bias.y}\n"
            f"  - {calib_params.f_bias.z}\n"
            f"t_bias:\n"
            f"  - {calib_params.t_bias.x}\n"
            f"  - {calib_params.t_bias.y}\n"
            f"  - {calib_params.t_bias.z}\n"
        )
        with open(filename, "w") as f:
            f.write(data_str)

    def save_sample_data(self, filename: str) -> None:
        """
        Save the sample data to a file

        Parameters:
            filename: Path to save the sample data
        """
        data_str = "\n".join(
            f"{' '.join(map(str, gravity))} {' '.join(map(str, ft_raw))}"
            for gravity, ft_raw in self._measurements
        )
        with open(filename, "w") as f:
            f.write(data_str)
