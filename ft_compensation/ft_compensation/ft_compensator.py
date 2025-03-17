import numpy as np


class FTCompensator:
    def __init__(self, calib_params: np.ndarray | None = None) -> None:
        if calib_params and calib_params.shape != (10,):
            raise ValueError("calib_params should be of shape (10,)")
        self.params = calib_params

    def load_calibration(self, calib_params: np.ndarray) -> None:
        if calib_params.shape != (10,):
            raise ValueError("calib_params should be of shape (10,)")
        self.params = calib_params

    def compensate(self, raw_ft: np.ndarray, gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compensate the raw force-torque measurement for gravity effects,
        returning both the true external force and torque.

        Parameters:
          raw_ft (np.ndarray): The raw force-torque measurement as a 6-element array,
                               ordered as [Fx, Fy, Fz, Mx, My, Mz] (Force in N, Torque in Nm).
          gravity (np.ndarray): The gravity vector as a 3-element array [gx, gy, gz] in m/s².

        Returns:
          (np.ndarray, np.ndarray): A tuple containing:
             - External force: 3-element array (N)
             - External torque: 3-element array (Nm)

        Physical model:
          Force:
              F_measured = -g * p0 + [p4, p5, p6] + F_external
              => F_external = F_measured + g * p0 - [p4, p5, p6]

          Torque:
              T_measured = g ✗ [p1, p2, p3] + [p7, p8, p9] + T_external
              => T_external = T_measured - (g ✗ [p1, p2, p3]+ [p7, p8, p9])
        """
        if self.params is None:
            raise ValueError("Calibration parameters have not been loaded.")
        if raw_ft.shape[0] < 6 or gravity.shape[0] < 3:
            raise ValueError("raw_ft must have 6 elements and gravity must have 3 elements.")

        # Compute the gravity contribution for force:
        F_external = raw_ft[:3] + gravity * self.params[0] - self.params[4:7]

        # Compute the gravity (and coupling) contribution for torque:
        T_external = raw_ft[3:6] - np.cross(gravity, self.params[1:4]) - self.params[7:10]

        return F_external, T_external


# ------------------ Example Usage ------------------
if __name__ == "__main__":
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

    compensator = FTCompensator()
    compensator.load_calibration(calib_params)

    # Example raw force-torque measurement:
    # Order: [Fx, Fy, Fz, Mx, My, Mz] (Force in N, Torque in Nm)
    raw_ft = np.array([-0.809967, -5.49514, 1.66939, -0.523497, 0.235421, -0.252396])

    # Example gravity vector (m/s²)
    gravity = np.array([-6.93651, -6.93688, 0.0241249])

    # Compute the compensated external force and torque
    F_external, T_external = compensator.compensate(raw_ft, gravity)
    print("External Force (N):", F_external)
    print("External Torque (Nm):", T_external)
