import sys
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

    def __init__(self, node: FTSamplerNode):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.calibrator = FTCalibrator()

        self.node = node
        self.node.set_sample_callback(self.sample_callback)
        self.ui.sampleButton.clicked.connect(lambda: self.node.trigger_sample())

        self.sample_received.connect(self._process_sample)

    def sample_callback(self, sample: SampleResult):
        self.sample_received.emit(sample)  # should return immediately

    def _process_sample(self, sample: SampleResult):
        self.ui.sampleList.addItem(self.sample_display_format(sample.ft_mean))
        self.calibrator.add_measurement(sample.gravity, sample.ft_mean)
        result = self.calibrator.get_calibration()
        self._update_result_display(result)

    def sample_display_format(self, ft_mean: np.ndarray):
        return (
            f"{strftime('%H:%M:%S', localtime())}\n"
            f"  Force: [{ft_mean[0]:.6f}, {ft_mean[1]:.6f}, {ft_mean[2]:.6f}]\n"
            f"  Torque: [{ft_mean[3]:.6f}, {ft_mean[4]:.6f}, {ft_mean[5]:.6f}]"
        )

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
    rclpy.init(args=args)

    sampler_node = FTSamplerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(sampler_node)
    thread = Thread(target=executor.spin)
    thread.start()

    app = QApplication(sys.argv)
    window = FTCalibrationGUI(sampler_node)

    try:
        window.show()
        sys.exit(app.exec())
    finally:
        sampler_node.get_logger().info("shutting down node")
        sampler_node.destroy_node()  # close node
        executor.shutdown()
        rclpy.shutdown()  # close rclpy
        thread.join()


if __name__ == "__main__":
    main()
