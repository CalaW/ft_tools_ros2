from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ft_bringup = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("ortho_bringup"), "launch", "net_ft_broadcaster.launch.py"]
        )
    )

    ur_driver = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
        ),
        launch_arguments={
            "robot_ip": "192.168.50.3",
            # "robot_ip": "192.168.56.101",
            "ur_type": "ur5",
            "description_launchfile": PathJoinSubstitution(
                [FindPackageShare("ortho_bringup"), "launch", "ortho_ur_rsp.launch.py"]
            ),
        }.items(),
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
