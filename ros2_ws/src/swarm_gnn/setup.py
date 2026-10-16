import os
from glob import glob

from setuptools import find_packages, setup

package_name = "swarm_gnn"

setup(
    name=package_name,
    version="1.0.0",
    # `training` is installed alongside `swarm_gnn` because the
    # train_allocator console script lives there and imports scenario_gen.
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "checkpoints"), glob("checkpoints/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="khan",
    maintainer_email="platformanalytics23@gmail.com",
    description="GNN task coordination for a heterogeneous drone/UGV/quadruped swarm.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gnn_allocator = swarm_gnn.nodes.gnn_allocator_node:main",
            "mission_executive = swarm_gnn.nodes.mission_executive_node:main",
            "rf_localizer = swarm_gnn.nodes.rf_localizer_node:main",
            "threat_field = swarm_gnn.nodes.threat_field_node:main",
            "agent_controller = swarm_gnn.nodes.agent_controller_node:main",
            "kinematic_platform = swarm_gnn.nodes.kinematic_platform_node:main",
            "swarm_viz = swarm_gnn.nodes.swarm_viz_node:main",
            "train_allocator = training.train_allocator:main",
        ],
    },
)
