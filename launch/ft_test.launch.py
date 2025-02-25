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

    ft_test_node = Node(
        package="ft_calib_comp",
        executable="ft_test.py",
        name="ft_test",
        output="screen",
    )

    return LaunchDescription([ft_bringup, ft_test_node])
