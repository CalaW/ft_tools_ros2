import sys
from pathlib import Path
from threading import Thread
from time import localtime, strftime

import numpy as np
import rclpy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from rclpy.executors import MultiThreadedExecutor

from ft_calibration import FTCalibrator, FTSamplerNode, Ui_MainWindow
from ft_calibration.ft_calibrator import CalibrationResult
from ft_calibration.ft_sampler_node import SampleResult


class FTCalibrationGUI(QMainWindow):
    sample_received = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.calibrator = FTCalibrator()
        self.node = None
        def spin_node():
            rclpy.init()
            self.node = FTSamplerNode()
            try:
                rclpy.spin(self.node)
            except KeyboardInterrupt:
                pass
            finally:
                self.node.destroy_node()
                rclpy.shutdown()
            
        Thread(target=spin_node, daemon=True).start()

        while self.node is None:
            pass

        self.node.set_sample_callback(self.sample_callback)
        self.ui.sampleButton.clicked.connect(lambda: self.node.trigger_sample())
        self.ui.saveButton.clicked.connect(lambda: self.save_calibration())

        self.sample_received.connect(self._process_sample)

    def sample_callback(self, sample: SampleResult):
        self.sample_received.emit(sample)  # should return immediately

    def _process_sample(self, sample: SampleResult):
        self.ui.sampleList.addItem(self.sample_display_format(sample.ft_mean, sample.gravity))
        self.calibrator.add_measurement(sample.gravity, sample.ft_mean)
        result = self.calibrator.get_calibration()
        self._update_result_display(result)

    def sample_display_format(self, ft_mean: np.ndarray, gravity: np.ndarray):
        return (
            f"\n{strftime('%H:%M:%S', localtime())}\n"
            f"  Force: [{ft_mean[0]:.6f}, {ft_mean[1]:.6f}, {ft_mean[2]:.6f}]\n"
            f"  Torque: [{ft_mean[3]:.6f}, {ft_mean[4]:.6f}, {ft_mean[5]:.6f}]\n"
            f"  Gravity: [{gravity[0]:.6f}, {gravity[1]:.6f}, {gravity[2]:.6f}]\n"
        )

    def save_calibration(self):
        # create .ros if not exist
        if not (Path.home() / ".ros").exists():
            (Path.home() / ".ros").mkdir()
        self.calibrator.save_calibration(Path.home() / ".ros" / "ft_calibration.yaml")
        self.calibrator.save_sample_data(Path.home() / ".ros" / "ft_calibration_samples.txt")

    def _update_result_display(self, result: CalibrationResult):
        self.ui.massLabel.setText(f"{result.mass:.3f}")
        self.ui.cogXLabel.setText(f"{result.cog.x:.3f}")
        self.ui.cogYLabel.setText(f"{result.cog.y:.3f}")
        self.ui.cogZLabel.setText(f"{result.cog.z:.3f}")
        self.ui.forceBiasXLabel.setText(f"{result.f_bias.x:.3f}")
        self.ui.forceBiasYLabel.setText(f"{result.f_bias.y:.3f}")
        self.ui.forceBiasZLabel.setText(f"{result.f_bias.z:.3f}")
        self.ui.torqueBiasXLabel.setText(f"{result.t_bias.x:.3f}")
        self.ui.torqueBiasYLabel.setText(f"{result.t_bias.y:.3f}")
        self.ui.torqueBiasZLabel.setText(f"{result.t_bias.z:.3f}")


def main(args=None):
    app = QApplication(sys.argv)
    window = FTCalibrationGUI()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
