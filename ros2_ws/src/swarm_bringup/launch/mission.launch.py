"""OPERATION SENTINEL GRID === the whole mission, in RViz.

    ros2 launch swarm_bringup mission.launch.py
    ros2 launch swarm_bringup mission.launch.py use_gnn:=false   # oracle baseline
    ros2 launch swarm_bringup mission.launch.py rviz:=false      # logs only

There is no physics engine. Each platform is integrated kinematically by
swarm_gnn/kinematic_platform against the sector geometry in
swarm_gnn/model/sector.py, which is also what the planner searches and what
swarm_viz draws, so the picture in RViz is exactly the world the robots are
navigating. The robots appear as their full articulated URDFs, with the
quadruped's gait, the UGV's wheels and the drone's rotors animated from the
same velocity command the coordination layer issued.

Arguments
    rviz        run RViz with the mission display (default true)
    use_gnn     false runs the analytic cost oracle instead of the network
    checkpoint  explicit GNN checkpoint path; empty auto-discovers
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# agent id -> (xacro path, platform kind)
PLATFORMS = {
    "drone_1": ("urdf/drone/drone.urdf.xacro", "drone"),
    "ugv_1": ("urdf/ugv/ugv.urdf.xacro", "ugv"),
    "dog_1": ("urdf/quadruped/quadruped.urdf.xacro", "quadruped"),
}


def _platform_nodes(context, *args, **kwargs):
    """Per robot: a description publisher and a kinematic platform."""
    import xacro

    desc_share = get_package_share_directory("swarm_description")
    actions = []

    for agent_id, (rel, kind) in PLATFORMS.items():
        robot_description = xacro.process_file(
            os.path.join(desc_share, rel),
            mappings={"namespace": agent_id, "tf_prefix": agent_id},
        ).toxml()

        actions.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=agent_id,
                name="robot_state_publisher",
                output="log",
                parameters=[{
                    "robot_description": robot_description,
                    "use_sim_time": False,
                    # Every link is published under <agent_id>/, so the three
                    # TF trees coexist beneath the shared `map` frame.
                    "frame_prefix": f"{agent_id}/",
                }],
            )
        )
        actions.append(
            Node(
                package="swarm_gnn",
                executable="kinematic_platform",
                namespace=agent_id,
                name="kinematic_platform",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    "agent_id": agent_id,
                    "platform": kind,
                    "use_sim_time": False,
                    "frame_prefix": f"{agent_id}/",
                }],
            )
        )
    return actions


def generate_launch_description():
    gnn_share = get_package_share_directory("swarm_gnn")
    bringup_share = get_package_share_directory("swarm_bringup")

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("use_gnn", default_value="true"),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(gnn_share, "config", "mission.yaml")),

        OpaqueFunction(function=_platform_nodes),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gnn_share, "launch", "swarm_layer.launch.py")),
            launch_arguments={
                "params_file": LaunchConfiguration("params_file"),
                "use_gnn": LaunchConfiguration("use_gnn"),
                "checkpoint": LaunchConfiguration("checkpoint"),
            }.items(),
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", os.path.join(bringup_share, "rviz", "mission.rviz")],
            parameters=[{"use_sim_time": False}],
        ),
    ])
