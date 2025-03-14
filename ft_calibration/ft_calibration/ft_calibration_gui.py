import sys
from threading import Thread

import rclpy
from PyQt5.QtWidgets import QApplication, QMainWindow
from rclpy.executors import MultiThreadedExecutor

from ft_calibration import FTSamplerNode, Ui_MainWindow


class FTCalibrationGUI(QMainWindow):
    def __init__(self, node: FTSamplerNode):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.node = node
        self.node.set_sample_callback(self.sample_callback)
        self.ui.sampleButton.clicked.connect(lambda: self.node.trigger_sample())

    def sample_callback(self, wtf):
        print(wtf)


def main(args=None):
    rclpy.init(args=args)

    calib_node = FTSamplerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(calib_node)
    thread = Thread(target=executor.spin)
    thread.start()

    app = QApplication(sys.argv)
    window = FTCalibrationGUI(calib_node)

    try:
        window.show()
        sys.exit(app.exec())
    finally:
        calib_node.get_logger().info("shutting down node")
        calib_node.destroy_node()  # close node
        executor.shutdown()
        rclpy.shutdown()  # close rclpy


if __name__ == "__main__":
    main()
