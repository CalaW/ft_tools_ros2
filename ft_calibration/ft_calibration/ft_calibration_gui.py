import sys
from threading import Thread
from time import localtime, strftime

import numpy as np
import rclpy
from PyQt5.QtWidgets import QApplication, QMainWindow
from rclpy.executors import MultiThreadedExecutor

from ft_calibration import FTCalibrator, FTSamplerNode, Ui_MainWindow
from ft_calibration.ft_calibrator import CalibrationResult


class FTCalibrationGUI(QMainWindow):
    def __init__(self, node: FTSamplerNode):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.calibrator = FTCalibrator()

        self.node = node
        self.node.set_sample_callback(self.sample_callback)
        self.ui.sampleButton.clicked.connect(lambda: self.node.trigger_sample())

    def sample_callback(self, sample):
        self.ui.sampleList.addItem(self.sample_display_format(1, sample))
        self.calibrator.add_measurement(np.array([0, 1, 0]), sample)
        result = self.calibrator.get_calibration()
        self.update_result_display(result)

    def sample_display_format(self, index, sample):
        return f"{strftime('%H:%M:%S', localtime())}\n  Force: [{sample[0]:.6f}, {sample[1]:.6f}, {sample[2]:.6f}]\n  Torque: [{sample[3]:.6f}, {sample[4]:.6f}, {sample[5]:.6f}]"

    def update_result_display(self, result: CalibrationResult):
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
