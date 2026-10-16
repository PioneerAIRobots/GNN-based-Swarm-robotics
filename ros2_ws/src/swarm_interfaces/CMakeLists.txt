cmake_minimum_required(VERSION 3.8)
project(swarm_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentState.msg"
  "msg/TaskSpec.msg"
  "msg/TaskArray.msg"
  "msg/Assignment.msg"
  "msg/AssignmentArray.msg"
  "msg/ThreatDetection.msg"
  "msg/ThreatEstimate.msg"
  "msg/ThreatArray.msg"
  "msg/GraphDebug.msg"
  "msg/MissionStatus.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
