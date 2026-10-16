"""QoS profiles, chosen per topic by what the consumer actually needs.

Getting these wrong is the usual cause of a multi-robot stack that works on one
machine and stalls on two, so each profile below says why it is what it is.
"""
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

#: Telemetry (odometry, agent state, detections). Best effort and shallow: a
#: dropped sample is replaced by a fresher one 0.1 s later, and queueing stale
#: poses actively harms a reactive controller.
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

#: Coordination decisions (tasks, assignments, mission status). Reliable and
#: transient-local: an agent that joins late, or restarts, must receive the
#: current assignment without waiting for the next tick.
COORDINATION_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

#: Fused world model (threat tracks, belief grid). Reliable, latched, slightly
#: deeper so a visualiser can catch up.
WORLD_MODEL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=3,
)
