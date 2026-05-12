# ft_tools_ros2

ROS 2 tools for calibrating and compensating force/torque (FT) sensors. This repository contains two packages that can be built with `colcon`.

## Packages

### `ft_calibration`
Python package for collecting samples and estimating FT sensor calibration parameters.

**Nodes**
- `ft_calibration_gui` (PyQt5 GUI)
- `ft_sampler_node` (sample collection node)

**Topics / frames**
- Subscribes to: `/ft/force_torque_sensor_broadcaster/wrench`
- Requires TF from `world` to `ati_measuring_face`
- Assumes gravity vector `[0, 0, -9.81]` m/s² in the `world` frame

**Outputs**
- `~/.ros/ft_calibration.yaml` (estimated mass, center of gravity, force/torque bias)
- `~/.ros/ft_calibration_samples.txt` (raw sample set)

**Launch**
- `ros2 launch ft_calibration ft_calibration.launch.py` (includes launch files from `net_ft_driver` and `dental_bringup`; these packages must be in your workspace)

### `ft_compensation`
Compensates raw FT measurements using calibration parameters and gravity.

**Nodes**
- `ft_compensation_node` (C++ passthrough node, republishes to `wrench_compensated`)
- `ft_compensation_node.py` (Python node that applies gravity compensation)

**Topics / frames (Python node)**
- Subscribes to: `/ft/force_torque_sensor_broadcaster/wrench`
- Publishes: `/wrench_compensated`
- Requires TF from `world` to `ati_measuring_face`

**Library**
- `FTCompensator` in `ft_compensation/ft_compensator.py` performs the gravity/bias compensation.

## Build

Replace `humble` with your ROS 2 distribution.

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select ft_calibration ft_compensation
source install/setup.bash
```

## Usage

### Calibration GUI
```bash
ros2 run ft_calibration ft_calibration_gui
```

### Sample-only node
```bash
ros2 run ft_calibration ft_sampler_node
```

### Compensation nodes
```bash
ros2 run ft_compensation ft_compensation_node
ros2 run ft_compensation ft_compensation_node.py
```

## Dependencies
- ROS 2 (`rclcpp`, `rclpy`)
- `geometry_msgs`
- `tf2_ros`
- `numpy`
- `PyQt5` (for the calibration GUI)
- `net_ft_driver` and `dental_bringup` (required to run `ft_calibration.launch.py`)

## License
Apache-2.0
