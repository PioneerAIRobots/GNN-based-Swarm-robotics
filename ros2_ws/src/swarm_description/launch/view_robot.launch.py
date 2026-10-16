"""Inspect a single platform in RViz, with sliders for its joints.

    ros2 launch swarm_description view_robot.launch.py robot:=quadruped

`robot` is one of drone | ugv | quadruped, with sliders for every joint.
Useful for checking a URDF in isolation: link frames, joint limits, and where
each sensor physically sits.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

_XACRO = {
    "drone": "urdf/drone/drone.urdf.xacro",
    "ugv": "urdf/ugv/ugv.urdf.xacro",
    "quadruped": "urdf/quadruped/quadruped.urdf.xacro",
}


def _setup(context, *args, **kwargs):
    import xacro

    share = get_package_share_directory("swarm_description")
    robot = LaunchConfiguration("robot").perform(context)
    if robot not in _XACRO:
        raise RuntimeError(f"robot must be one of {sorted(_XACRO)}, got '{robot}'")

    path = os.path.join(share, _XACRO[robot])
    doc = xacro.process_file(path, mappings={"namespace": robot, "tf_prefix": robot})
    description = doc.toxml()

    rviz_config = os.path.join(share, "rviz", "view_robot.rviz")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": description, "use_sim_time": False}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", rviz_config] if os.path.exists(rviz_config) else [],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot",
                default_value="quadruped",
                description="Platform to inspect: drone | ugv | quadruped",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
