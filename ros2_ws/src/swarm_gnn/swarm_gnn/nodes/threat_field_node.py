#!/usr/bin/env python3
"""Simulates the RF and CBRN phenomenology the mission turns on.

The two sensors that decide this mission === the DF arrays and the quadruped's
point sampler === measure fields, not geometry, so no amount of simulator
fidelity elsewhere would produce them. This node supplies them: it holds the
ground-truth emitter table, reads each robot's true pose, and publishes the
observation that robot's array would actually have produced, including the
failure modes that make the problem hard:

  * log-normal shadowing on received power (6.5 dB), so a single RSSI says
    almost nothing about range;
  * per-platform bearing noise set by that platform's DF baseline, taken from
    the same numbers as the URDF's rf_df_array macro;
  * occlusion by buildings, which both attenuates the signal and biases the
    bearing, so a robot behind a block gets a plausible but wrong cut;
  * a sensitivity floor, below which a platform simply hears nothing.

It also models what the emitters do back: an active jammer degrades the comms
quality and GNSS of any robot inside its radius, which is what forces the team
onto the relay task and off absolute positioning.

This node is the simulator's half of the contract. Nothing in the coordination
stack reads ground truth; it only ever sees the noisy ThreatDetection stream.
"""
from __future__ import annotations

import math
import random

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String
from swarm_interfaces.msg import ThreatDetection

from ..model import sector
from ..model.rf_localizer import SHADOWING_SIGMA_DB, expected_rssi, wrap_angle
from ..model.spec import PROFILES, Platform
from ..util.qos import SENSOR_QOS, WORLD_MODEL_QOS

#: Emitter classes, matching ThreatDetection CLASS_*.
CLASS_UNKNOWN, CLASS_JAMMER, CLASS_IED_TRIGGER, CLASS_CIVILIAN = 0, 1, 2, 3

_CLASS_BY_NAME = {
    "jammer": CLASS_JAMMER,
    "ied_trigger": CLASS_IED_TRIGGER,
    "civilian": CLASS_CIVILIAN,
}


class EmitterState:
    """Runtime wrapper around a `sector.Emitter`: duty cycle and liveness.

    The emitter table itself lives in `model/sector.py`. It used to be
    duplicated into nine parallel ROS parameter arrays here and into a
    generated Gazebo world, and the copies drifted: at one point every emitter
    sat inside a building volume, where no thermal confirm and no CBRN sample
    is physically possible. One source of truth, validated by
    `sector.validate()`.
    """

    def __init__(self, spec: sector.Emitter) -> None:
        self.spec = spec
        self.neutralized = False
        self._keyed = True

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def position(self) -> tuple[float, float, float]:
        return self.spec.position

    @property
    def emitter_class(self) -> int:
        return _CLASS_BY_NAME.get(self.spec.kind, CLASS_UNKNOWN)

    def step(self, rng: random.Random) -> None:
        if self.neutralized:
            self._keyed = False
            return
        # Bernoulli keying per tick: simple, and it reproduces the operational
        # reality that an emitter heard once may be silent on arrival.
        self._keyed = rng.random() < self.spec.duty_cycle

    @property
    def active(self) -> bool:
        return self._keyed and not self.neutralized


class ThreatFieldNode(Node):
    def __init__(self) -> None:
        super().__init__("threat_field")

        self.declare_parameter("update_rate", 4.0)
        self.declare_parameter("agent_namespaces", ["drone_1", "ugv_1", "dog_1"])
        self.declare_parameter("agent_platforms", ["drone", "ugv", "quadruped"])
        self.declare_parameter("occlusion_loss_db", 11.0)
        self.declare_parameter("random_seed", 7)

        self._rng = random.Random(int(self.get_parameter("random_seed").value))

        problems = sector.validate()
        if problems:
            # Refuse to run an incoherent layout rather than produce a mission
            # whose emitters cannot be reached or observed.
            for problem in problems:
                self.get_logger().error(f"layout: {problem}")
            raise RuntimeError(f"{len(problems)} sector layout problem(s)")

        self.emitters = [EmitterState(e) for e in sector.EMITTERS]

        platform_by_name = {
            "drone": Platform.DRONE,
            "ugv": Platform.UGV,
            "quadruped": Platform.QUADRUPED,
            "dog": Platform.QUADRUPED,
        }
        namespaces = list(self.get_parameter("agent_namespaces").value)
        platforms = list(self.get_parameter("agent_platforms").value)
        if len(namespaces) != len(platforms):
            raise RuntimeError("agent_namespaces and agent_platforms must be the same length")

        self.agents: dict[str, dict] = {}
        for ns, plat_name in zip(namespaces, platforms):
            plat = platform_by_name.get(plat_name.lower())
            if plat is None:
                raise RuntimeError(f"unknown platform '{plat_name}' for namespace '{ns}'")
            self.agents[ns] = {"platform": plat, "pose": None}
            # Ground truth: this node *is* the simulator, so reading the true
            # pose here is correct. The coordination stack never subscribes.
            self.create_subscription(
                Odometry, f"/{ns}/ground_truth",
                self._make_pose_cb(ns), SENSOR_QOS,
            )

        self.detection_pub = self.create_publisher(
            ThreatDetection, "/swarm/detections", SENSOR_QOS
        )
        # Per-agent environmental effects the emitters inflict back on the team.
        self.comms_pubs = {
            ns: self.create_publisher(Float32, f"/{ns}/comms_quality", SENSOR_QOS)
            for ns in self.agents
        }
        self.cbrn_pubs = {
            ns: self.create_publisher(Float32, f"/{ns}/cbrn_reading", SENSOR_QOS)
            for ns in self.agents
        }
        # The executive reports a neutralisation here; the emitter then goes
        # quiet, which is what the BDA pass is supposed to detect.
        self.create_subscription(
            String, "/swarm/neutralize", self._on_neutralize, WORLD_MODEL_QOS
        )

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(0.5, rate), self._tick)

        self.get_logger().info(
            f"threat_field up: {len(self.emitters)} emitters "
            f"({sum(1 for e in self.emitters if e.kind == 'civilian')} civilian decoy), "
            f"{len(self.agents)} observers, {len(sector.BUILDINGS)} occluders"
        )

    # ------------------------------------------------------------------ setup
    def _make_pose_cb(self, ns: str):
        def cb(msg: Odometry) -> None:
            p = msg.pose.pose.position
            self.agents[ns]["pose"] = (p.x, p.y, p.z)
        return cb

    # ------------------------------------------------------------- physics
    def _segment_occlusion(self, a, b) -> int:
        """Buildings on the a-b ray, from the shared sector geometry."""
        return sector.segment_occlusions(a, b)

    def _observe(self, ns: str, agent: dict, emitter: EmitterState) -> ThreatDetection | None:
        """Generate one platform's observation of one emitter, or None."""
        pos = agent["pose"]
        profile = PROFILES[agent["platform"]]

        planar = math.dist(pos[:2], emitter.position[:2])
        slant = math.sqrt(planar**2 + (pos[2] - emitter.position[2]) ** 2)

        occlusions = self._segment_occlusion(pos, emitter.position)
        extra_loss = occlusions * float(self.get_parameter("occlusion_loss_db").value)

        rssi = (
            expected_rssi(emitter.spec.power_dbm, slant, extra_loss_db=extra_loss)
            + self._rng.gauss(0.0, SHADOWING_SIGMA_DB)
        )

        # Sensitivity floor: the array simply does not hear it.
        sensitivity = {
            Platform.DRONE: -104.0, Platform.UGV: -112.0, Platform.QUADRUPED: -108.0
        }[agent["platform"]]
        if rssi < sensitivity:
            return None

        true_bearing = math.atan2(
            emitter.position[1] - pos[1], emitter.position[0] - pos[0]
        )
        sigma = profile.df_sigma
        # Occlusion does not merely add noise, it adds *bias*: the array locks
        # onto a reflected path, so the cut is confidently wrong. This is the
        # error the Huber weighting and the multi-modal grid exist to survive.
        bias = 0.0
        if occlusions:
            sigma *= 1.0 + 0.8 * occlusions
            bias = self._rng.gauss(0.0, 0.30 * occlusions)

        # A weak signal also degrades the bearing estimate, as SNR sets the
        # achievable angular resolution for a given baseline.
        snr_margin = rssi - sensitivity
        if snr_margin < 15.0:
            sigma *= 1.0 + (15.0 - snr_margin) / 15.0

        msg = ThreatDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.observer_id = ns
        msg.observer_position = Point(x=pos[0], y=pos[1], z=pos[2])
        msg.rssi_dbm = float(rssi)
        msg.bearing_rad = float(wrap_angle(true_bearing + bias + self._rng.gauss(0.0, sigma)))
        msg.bearing_sigma_rad = float(sigma)
        msg.frequency_mhz = emitter.spec.freq_mhz
        # The *true* class is published, but the coordination stack treats it as
        # unknown until the confirm tasks corroborate it; see mission_executive.
        msg.emitter_class = emitter.emitter_class
        return msg

    # ---------------------------------------------------------------- tick
    def _tick(self) -> None:
        for emitter in self.emitters:
            emitter.step(self._rng)

        for ns, agent in self.agents.items():
            if agent["pose"] is None:
                continue

            # ---- RF detections -------------------------------------------
            for emitter in self.emitters:
                if not emitter.active:
                    continue
                det = self._observe(ns, agent, emitter)
                if det is not None:
                    self.detection_pub.publish(det)

            # ---- jamming effect on this agent ----------------------------
            quality = 1.0
            for emitter in self.emitters:
                if not emitter.active or emitter.spec.jam_radius <= 0.0:
                    continue
                d = math.dist(agent["pose"][:2], emitter.position[:2])
                if d < emitter.spec.jam_radius:
                    # Linear roll-off from total denial at the mast to clear at
                    # the edge of the jam radius.
                    quality = min(quality, max(0.05, d / emitter.spec.jam_radius))
            self.comms_pubs[ns].publish(Float32(data=float(quality)))

            # ---- CBRN plume reading --------------------------------------
            # Only the IED triggers have a chemical signature (binary charge
            # precursor). The reading falls off as 1/d^2 from the site and is
            # only meaningful within the sampler's standoff.
            reading = 0.0
            for emitter in self.emitters:
                if emitter.kind != "ied_trigger" or emitter.neutralized:
                    continue
                d = math.dist(agent["pose"][:2], emitter.position[:2])
                reading += 1.4 / (1.0 + d * d / 4.0)
            reading += abs(self._rng.gauss(0.0, 0.012))
            self.cbrn_pubs[ns].publish(Float32(data=float(min(1.0, reading))))

    def _on_neutralize(self, msg: String) -> None:
        """A marker has been emplaced: silence the nearest live emitter.

        Matching by position rather than by id is deliberate === the team only
        ever knows an estimated position, so this is also the check on whether
        they marked the right thing. The message payload is "x,y".
        """
        try:
            x_str, y_str = msg.data.split(",")
            x, y = float(x_str), float(y_str)
        except ValueError:
            self.get_logger().warn(f"malformed neutralize request: '{msg.data}'")
            return

        best, best_d = None, float("inf")
        for emitter in self.emitters:
            if emitter.neutralized:
                continue
            d = math.dist((x, y), emitter.position[:2])
            if d < best_d:
                best, best_d = emitter, d

        # 6 m is the marker's effective radius.
        if best is not None and best_d <= 6.0:
            best.neutralized = True
            verdict = "FRATRICIDE: civilian transmitter" if best.kind == "civilian" else "valid"
            self.get_logger().warn(
                f"emitter '{best.name}' ({best.kind}) neutralised at {best_d:.1f} m "
                f"from the marker === {verdict}"
            )
        else:
            self.get_logger().warn(
                f"marker at ({x:.1f}, {y:.1f}) hit nothing; nearest live emitter "
                f"is {best_d:.1f} m away"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThreatFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
