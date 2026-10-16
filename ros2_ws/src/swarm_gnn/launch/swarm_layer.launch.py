"""The coordination layer: allocator, executive, localiser, threat field, viz.

Included by swarm_bringup's mission.launch.py. Kept separate so the
coordination stack can be brought up against a different platform layer
without touching mission logic.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

AGENTS = [
    ("drone_1", "drone"),
    ("ugv_1", "ugv"),
    ("dog_1", "quadruped"),
]


def _default_checkpoint() -> str:
    """Prefer the installed checkpoint, then the source tree.

    Returning "" is a valid outcome: the allocator then runs on the analytic
    cost oracle, says so in its log, and the mission still flies.
    """
    candidates = [
        os.path.join(
            get_package_share_directory("swarm_gnn"), "checkpoints", "allocator.pt"
        ),
        os.path.join(os.getcwd(), "src", "swarm_gnn", "checkpoints", "allocator.pt"),
        os.path.join(os.getcwd(), "checkpoints", "allocator.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _setup(context, *args, **kwargs):
    params = LaunchConfiguration("params_file").perform(context)
    use_gnn = LaunchConfiguration("use_gnn").perform(context).lower() == "true"
    checkpoint = LaunchConfiguration("checkpoint").perform(context) or _default_checkpoint()

    # `use_sim_time` is false throughout: there is no clock publisher, the
    # platforms integrate against wall time, and a params file would otherwise
    # pin it (a params file takes precedence over a launch dict).
    common = [params, {"use_sim_time": False}]

    nodes = [
        Node(package="swarm_gnn", executable="gnn_allocator", name="gnn_allocator",
             output="screen", emulate_tty=True,
             parameters=common + [{"use_gnn": use_gnn, "checkpoint": checkpoint}]),
        Node(package="swarm_gnn", executable="mission_executive",
             name="mission_executive", output="screen", emulate_tty=True,
             parameters=common),
        Node(package="swarm_gnn", executable="rf_localizer", name="rf_localizer",
             output="screen", emulate_tty=True, parameters=common),
        Node(package="swarm_gnn", executable="threat_field", name="threat_field",
             output="screen", emulate_tty=True, parameters=common),
        Node(package="swarm_gnn", executable="swarm_viz", name="swarm_viz",
             output="screen", parameters=common),
    ]

    # One controller per platform, in that platform's namespace so its
    # cmd_vel / odom / scan resolve to the robot's own topics.
    for agent_id, platform in AGENTS:
        nodes.append(
            Node(package="swarm_gnn", executable="agent_controller",
                 name="agent_controller", namespace=agent_id,
                 output="screen", emulate_tty=True,
                 parameters=common + [{"agent_id": agent_id, "platform": platform}])
        )
    return nodes


def generate_launch_description():
    share = get_package_share_directory("swarm_gnn")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(share, "config", "mission.yaml")),
        DeclareLaunchArgument(
            "use_gnn", default_value="true",
            description="false runs the analytic cost oracle instead of the network"),
        DeclareLaunchArgument(
            "checkpoint", default_value="",
            description="GNN checkpoint; empty auto-discovers, then falls back to the oracle"),
        OpaqueFunction(function=_setup),
    ])
