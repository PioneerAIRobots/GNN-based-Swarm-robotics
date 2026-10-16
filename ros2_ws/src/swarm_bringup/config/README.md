Mission parameters live in `swarm_gnn/config/mission.yaml`, which this package's
launch files pass through as `params_file`. Override on the command line:

    ros2 launch swarm_bringup mission.launch.py params_file:=/path/to/my.yaml
