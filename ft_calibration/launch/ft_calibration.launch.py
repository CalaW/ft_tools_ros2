from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ft_bringup = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("net_ft_driver"), "launch", "net_ft_broadcaster.launch.py"]
        )
    )

    ur_driver = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("dental_bringup"), "launch", "ur_control.launch.py"]
        )
    )

    ft_calibration_gui = Node(
        package="ft_calibration",
        executable="ft_calibration_gui",
        name="ft_calibration_gui",
        output="screen",
    )

    return LaunchDescription(
        [
            ft_bringup,
            ft_calibration_gui,
            ur_driver,
        ]
    )
