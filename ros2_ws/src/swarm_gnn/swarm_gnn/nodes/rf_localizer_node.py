#!/usr/bin/env python3
"""Fuses the swarm's DF detections into a shared threat picture.

Runs the two-stage estimator from model/rf_localizer.py over the pooled
detection stream and maintains a track per emitter hypothesis. Publishes:

  /swarm/threats       the fused tracks the executive and allocator act on
  /swarm/belief_grid   the posterior as an OccupancyGrid, for RViz

On centralisation
=================
One node fuses for the whole team here. That is a simulation convenience, not
an architectural assumption: the update is additive in log space and therefore
order-independent, so running a copy per robot over gossiped detections
converges to the same posterior. The topic layout already supports that === each
robot would subscribe to /swarm/detections and publish its own /ns/threats.
What would change is only the gossip protocol, which is out of scope here.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from swarm_interfaces.msg import ThreatArray, ThreatDetection, ThreatEstimate

from ..model.rf_localizer import (
    BayesianRFGrid,
    Detection,
    ThreatTrack,
    multilaterate,
    observer_baseline,
)
from ..util.conversions import detection_msg_to_model, track_to_estimate
from ..util.qos import SENSOR_QOS, WORLD_MODEL_QOS

#: ThreatDetection CLASS_* values, for log messages.
_CLASS_NAMES = {0: "unknown", 1: "JAMMER", 2: "IED_TRIGGER", 3: "civilian(no-strike)"}


class RFLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("rf_localizer")

        self.declare_parameter("update_rate", 2.0)
        self.declare_parameter("sector_half_extent", 60.0)
        self.declare_parameter("grid_resolution", 1.5)
        self.declare_parameter("tx_power_prior_dbm", 22.0)
        # Prior uncertainty on that power. Do not shrink this to make the
        # filter look confident: see BayesianRFGrid.__init__ on why a narrow
        # value produces confident fixes in the sector corners.
        self.declare_parameter("tx_power_sigma_db", 9.0)
        # Association gate: a detection-derived mode within this distance of an
        # existing track updates it rather than spawning a new one.
        self.declare_parameter("association_radius", 14.0)
        # A mode needs this much posterior mass before it becomes a track at all.
        self.declare_parameter("candidate_mass_threshold", 0.012)
        # ...and it must be supported by cuts from at least this many distinct
        # observers. A single platform cannot localise an emitter no matter how
        # many samples it takes (the range is unobservable from one point), so
        # promoting a one-observer mode to a track just manufactures work for
        # the team. Requiring two is the cheapest possible guard against that.
        self.declare_parameter("min_observers_for_candidate", 2)
        # Width of the grid-boundary band in which peaks are discarded as
        # truncation artifacts rather than emitters. See BayesianRFGrid.modes.
        self.declare_parameter("edge_margin", 6.0)
        # Promotion to LOCALIZED: the refined fix must be at least this tight.
        self.declare_parameter("localized_sigma", 4.0)
        # ...and the cuts must span at least this baseline (see multilaterate).
        self.declare_parameter("min_baseline", 10.0)
        self.declare_parameter("belief_decay", 0.992)
        self.declare_parameter("detection_memory", 240)
        # Channel width for grouping detections, MHz. Must be finer than the
        # spacing between emitters (0.5 MHz here) and coarser than the
        # receiver's frequency error.
        self.declare_parameter("channel_width_mhz", 0.4)

        # One belief grid per frequency channel, not one grid for the sector.
        #
        # A single shared grid mixes every emitter's bearing fans, and fans
        # from different emitters intersect at points where nothing is
        # transmitting. That produced tracks the team then spent a thermal
        # confirm and a CBRN sample investigating === real effort on a
        # phantom. A DF receiver measures frequency, `ThreatDetection` already
        # carries it, and the emitters here are 0.5 MHz or more apart, so
        # binning by channel separates them at source instead of trying to
        # suppress the consequences downstream.
        self.grids: dict[int, BayesianRFGrid] = {}
        self.tracks: dict[str, ThreatTrack] = {}
        self.detections: list[Detection] = []
        self._next_track_id = 0

        self.create_subscription(
            ThreatDetection, "/swarm/detections", self._on_detection, SENSOR_QOS
        )
        # Confirmations arrive from the agents as "<threat_id>:<modality>".
        self.create_subscription(
            String, "/swarm/confirmations", self._on_confirmation, WORLD_MODEL_QOS
        )
        self.create_subscription(
            String, "/swarm/neutralize", self._on_neutralized, WORLD_MODEL_QOS
        )

        self.threat_pub = self.create_publisher(ThreatArray, "/swarm/threats", WORLD_MODEL_QOS)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, "/swarm/belief_grid", WORLD_MODEL_QOS
        )

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(0.5, rate), self._tick)
        self._grid_publish_counter = 0

        probe = self._grid_for(0)
        self.get_logger().info(
            f"rf_localizer up: per-channel belief grids, "
            f"{probe.shape[0]}x{probe.shape[1]} at {probe.resolution} m, "
            f"{float(self.get_parameter('channel_width_mhz').value)} MHz channels"
        )

    def _channel(self, frequency_mhz: float) -> int:
        width = max(0.05, float(self.get_parameter("channel_width_mhz").value))
        return int(round(frequency_mhz / width))

    def _grid_for(self, channel: int) -> BayesianRFGrid:
        grid = self.grids.get(channel)
        if grid is None:
            grid = BayesianRFGrid(
                half_extent=float(self.get_parameter("sector_half_extent").value),
                resolution=float(self.get_parameter("grid_resolution").value),
                tx_power_prior_dbm=float(self.get_parameter("tx_power_prior_dbm").value),
                tx_power_sigma_db=float(self.get_parameter("tx_power_sigma_db").value),
            )
            self.grids[channel] = grid
        return grid

    # ------------------------------------------------------------- callbacks
    def _on_detection(self, msg: ThreatDetection) -> None:
        stamp = self.get_clock().now().nanoseconds * 1e-9
        det = detection_msg_to_model(msg, stamp)
        self._grid_for(self._channel(det.frequency_mhz)).update(det)
        self.detections.append(det)

        limit = int(self.get_parameter("detection_memory").value)
        if len(self.detections) > limit:
            # Drop the oldest: an emitter that moved or stopped should not be
            # held in place by hour-old cuts.
            self.detections = self.detections[-limit:]

    def _on_confirmation(self, msg: String) -> None:
        """A sensor modality reported on a track.

        Payload: ``<threat_id>:<modality>:<evidence>[,<evidence>...]``

        The reporter states what it *observed*, not what it concluded. The
        class is then derived from the pooled evidence by
        `ThreatTrack.classify`, because no single sensor distinguishes a
        jammer from a licensed transmitter, and letting each robot assert a
        class is what previously mislabelled the jammer as no-strike.

        Each distinct modality counts once towards corroboration: three
        thermal looks are still one line of evidence.
        """
        parts = msg.data.split(":")
        if len(parts) < 3:
            self.get_logger().warn(f"malformed confirmation: '{msg.data}'")
            return
        threat_id, modality, evidence_field = parts[0], parts[1], parts[2]
        track = self.tracks.get(threat_id)
        if track is None:
            self.get_logger().warn(f"confirmation for unknown track '{threat_id}'")
            return

        tokens = {t for t in evidence_field.split(",") if t}
        new_modality = modality not in track.modalities
        track.modalities.add(modality)
        track.evidence |= tokens
        track.confirm_count = len(track.modalities)

        previous_class = track.emitter_class
        track.emitter_class = track.classify()

        if new_modality or track.emitter_class != previous_class:
            self.get_logger().info(
                f"{threat_id} + {modality} {sorted(tokens)} -> "
                f"class {_CLASS_NAMES.get(track.emitter_class, '?')}, "
                f"{track.confirm_count}/2 modalities"
            )

        if track.confirm_count >= 2 and track.state < ThreatEstimate.STATE_CONFIRMED:
            track.state = ThreatEstimate.STATE_CONFIRMED
            self.get_logger().info(
                f"{threat_id} CONFIRMED as "
                f"{_CLASS_NAMES.get(track.emitter_class, '?')} at "
                f"({track.position[0]:.1f}, {track.position[1]:.1f}) "
                f"on evidence {sorted(track.evidence)}"
            )

    def _on_neutralized(self, msg: String) -> None:
        try:
            x, y = (float(v) for v in msg.data.split(","))
        except ValueError:
            return
        for track in self.tracks.values():
            if math.dist(track.position, (x, y)) <= 6.0:
                track.state = ThreatEstimate.STATE_NEUTRALIZED

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        decay = float(self.get_parameter("belief_decay").value)
        for grid in self.grids.values():
            grid.decay(decay)
        self._update_tracks()
        self._merge_tracks()
        self._publish_threats()

        # The grid is 80x80 floats; publishing it at 2 Hz is wasteful when only
        # a visualiser reads it.
        self._grid_publish_counter += 1
        if self._grid_publish_counter % 4 == 0:
            self._publish_grid()

    def _update_tracks(self) -> None:
        assoc_radius = float(self.get_parameter("association_radius").value)
        mass_threshold = float(self.get_parameter("candidate_mass_threshold").value)
        min_baseline = float(self.get_parameter("min_baseline").value)
        localized_sigma = float(self.get_parameter("localized_sigma").value)
        now = self.get_clock().now().nanoseconds * 1e-9

        min_observers = int(self.get_parameter("min_observers_for_candidate").value)
        edge_margin = float(self.get_parameter("edge_margin").value)

        # Modes are extracted per channel, so a peak is only ever formed from
        # detections that could have come from the same transmitter.
        modes: list[tuple[tuple[float, float], float, int]] = []
        for channel, grid in self.grids.items():
            for (mx, my), mass in grid.modes(
                min_separation=assoc_radius,
                # One channel carries one transmitter, so there is at most one
                # emitter to find per grid. Allowing three produced up to
                # twelve tracks for four emitters once the team spread out and
                # the extra cuts created more spurious intersections, and each
                # phantom re-opened confirm and sample tasks that kept the
                # mission from ever finishing. Two leaves room for a genuine
                # co-channel pair without inviting that.
                max_modes=2,
                mass_threshold=mass_threshold,
                edge_margin=edge_margin,
            ):
                modes.append(((mx, my), mass, channel))
        # Strongest first, so the best-supported hypotheses claim their tracks.
        modes.sort(key=lambda m: -m[1])

        for (mx, my), mass, channel in modes:
            track = self._associate((mx, my), assoc_radius, channel=channel)
            supporting = self._detections_pointing_at((mx, my), channel=channel)
            distinct_observers = {d.observer_id for d in supporting}

            if track is None and self._near_neutralized((mx, my), assoc_radius):
                # Detections already in memory keep a neutralised emitter's
                # grid peak alive for a while after it goes silent. Spawning a
                # fresh track there sends the team back to re-confirm and
                # re-sample something it has already dealt with === observed as
                # a third track appearing on top of a marked trigger.
                continue

            if track is None:
                if len(distinct_observers) < min_observers:
                    # Not enough independent looks yet; the mode stays in the
                    # grid (where the next detection can reinforce it) but does
                    # not become a track the executive will task against.
                    continue
                track = ThreatTrack(
                    threat_id=f"threat_{self._next_track_id}",
                    position=(mx, my),
                    channel=channel,
                )
                self._next_track_id += 1
                self.tracks[track.threat_id] = track
                self.get_logger().info(
                    f"{track.threat_id} CANDIDATE at ({mx:.1f}, {my:.1f}), mass {mass:.3f}"
                )

            if track.state == ThreatEstimate.STATE_NEUTRALIZED:
                continue

            track.confidence = float(min(1.0, mass))
            track.last_update = now

            # Detections plausibly belonging to this mode: the ones whose
            # bearing actually points at it. Pooling every detection in the
            # sector would mix emitters together and the refinement would
            # converge on the midpoint between two of them.
            relevant = supporting
            track.detections = relevant
            track.baseline_positions = [d.position[:2] for d in relevant if d.has_bearing]
            if relevant:
                track.frequency_mhz = float(
                    np.median([d.frequency_mhz for d in relevant])
                )

            refined = multilaterate(relevant, (mx, my), min_baseline=min_baseline)
            if refined is not None:
                fix, cov, rms = refined
                # Reject a refinement that has wandered far from the grid mode:
                # the grid is the robust estimate and disagreement that large
                # means the cut set is mixed, not that the fix is sharper.
                if math.dist(fix, (mx, my)) <= assoc_radius:
                    track.position = fix
                    track.covariance = cov
                else:
                    track.position = (mx, my)
                    track.covariance = self.grids[channel].covariance_at((mx, my))
            else:
                track.position = (mx, my)
                track.covariance = self.grids[channel].covariance_at((mx, my))

            if (
                track.state == ThreatEstimate.STATE_CANDIDATE
                and track.position_sigma <= localized_sigma
                and track.baseline_spread() >= min_baseline
            ):
                track.state = ThreatEstimate.STATE_LOCALIZED
                self.get_logger().info(
                    f"{track.threat_id} LOCALIZED at ({track.position[0]:.1f}, "
                    f"{track.position[1]:.1f}) sigma {track.position_sigma:.1f} m, "
                    f"baseline {track.baseline_spread():.0f} m"
                )

    def _merge_tracks(self) -> None:
        """Fuse tracks that have converged onto the same emitter.

        Association happens at creation time against a mode's *grid* position,
        but both tracks then get refined by multilateration and can walk
        together afterwards. A live run produced three tracks on the single
        emitter at (14, 38) === created 16 m apart, refined to within 10 m ===
        and the executive duly raised a confirm and a CBRN task for each,
        spending the quadruped's time twice on one target.

        The surviving track is the better-localised one, and it inherits the
        other's corroborations: evidence about an emitter should not be lost
        because the estimator briefly thought it was looking at two.
        """
        radius = float(self.get_parameter("association_radius").value)
        live = [
            t for t in self.tracks.values()
            if t.state != ThreatEstimate.STATE_NEUTRALIZED
        ]
        # Tightest sigma first, so it is the one that absorbs the others.
        live.sort(key=lambda t: t.position_sigma)

        absorbed: set[str] = set()
        for i, keep in enumerate(live):
            if keep.threat_id in absorbed:
                continue
            for other in live[i + 1:]:
                if other.threat_id in absorbed:
                    continue
                if keep.channel != other.channel:
                    # Different transmitters. Their fixes being close is a
                    # coincidence of geometry, not evidence they are one site.
                    continue
                if math.dist(keep.position, other.position) > radius:
                    continue

                keep.modalities |= other.modalities
                keep.evidence |= other.evidence
                keep.confirm_count = len(keep.modalities)
                keep.confidence = max(keep.confidence, other.confidence)
                # Re-derive from the pooled evidence rather than picking one
                # track's verdict: the two may hold complementary halves of it.
                keep.emitter_class = keep.classify()
                keep.state = max(keep.state, other.state)
                if keep.confirm_count >= 2 and keep.state < ThreatEstimate.STATE_CONFIRMED:
                    keep.state = ThreatEstimate.STATE_CONFIRMED

                absorbed.add(other.threat_id)
                self.get_logger().info(
                    f"merged {other.threat_id} into {keep.threat_id} "
                    f"({math.dist(keep.position, other.position):.1f} m apart, "
                    f"modalities {sorted(keep.modalities) or 'none'}, "
                    f"class {_CLASS_NAMES.get(keep.emitter_class, '?')})"
                )

        for tid in absorbed:
            self.tracks.pop(tid, None)

    def _near_neutralized(self, point: tuple[float, float], radius: float) -> bool:
        return any(
            t.state == ThreatEstimate.STATE_NEUTRALIZED
            and math.dist(t.position, point) <= radius
            for t in self.tracks.values()
        )

    def _associate(
        self, point: tuple[float, float], radius: float, channel: int | None = None
    ) -> ThreatTrack | None:
        """Nearest live track to `point`, preferring the same channel.

        The channel gate is widened rather than tightened: a mode on a channel
        that already has a track is almost certainly that same transmitter
        seen with a slightly different fix, so it associates out to a larger
        radius than a cross-channel candidate would. Without this, one emitter
        accumulated several tracks as its fix wandered, and the team was sent
        to re-confirm the same site repeatedly.
        """
        best, best_d = None, radius
        same_channel_radius = radius * 2.5
        for track in self.tracks.values():
            if track.state == ThreatEstimate.STATE_NEUTRALIZED:
                continue
            gate = (
                same_channel_radius
                if channel is not None and track.channel == channel
                else radius
            )
            d = math.dist(track.position, point)
            if d < gate and d < max(best_d, gate):
                if best is None or d < math.dist(best.position, point):
                    best, best_d = track, d
        return best

    def _detections_pointing_at(
        self, point: tuple[float, float], gate_sigmas: float = 3.0,
        channel: int | None = None,
    ) -> list[Detection]:
        """Detections consistent with an emitter at `point`, on one channel.

        Both gates matter. The bearing gate rejects cuts pointing elsewhere;
        the channel gate rejects cuts from a *different transmitter* that
        happen to point the same way, which is what let two emitters pool
        their detections and drag a fix to the midpoint between them.
        """
        out = []
        for det in self.detections:
            if not det.has_bearing:
                continue
            if channel is not None and self._channel(det.frequency_mhz) != channel:
                continue
            expected = math.atan2(point[1] - det.position[1], point[0] - det.position[0])
            resid = abs(
                math.atan2(
                    math.sin(det.bearing_rad - expected),
                    math.cos(det.bearing_rad - expected),
                )
            )
            if resid <= gate_sigmas * max(det.bearing_sigma_rad, 0.02):
                out.append(det)
        return out

    # -------------------------------------------------------------- publish
    def _publish_threats(self) -> None:
        msg = ThreatArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.threats = [track_to_estimate(t) for t in self.tracks.values()]
        self.threat_pub.publish(msg)

    def _publish_grid(self) -> None:
        if not self.grids:
            return
        # The operator wants one picture, so take the per-cell maximum across
        # channels: a cell is interesting if *any* transmitter might be there.
        posterior = None
        for grid in self.grids.values():
            p = grid.posterior()
            posterior = p if posterior is None else np.maximum(posterior, p)
        reference = next(iter(self.grids.values()))
        # Scale to 0..100 against the peak: the absolute probability of a cell
        # is tiny on an 80x80 grid, so an absolute scale renders as all-zero.
        peak = posterior.max()
        if peak <= 0.0:
            return
        scaled = np.clip((posterior / peak) * 100.0, 0, 100).astype(np.int8)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = reference.resolution
        grid_msg.info.width = reference.shape[0]
        grid_msg.info.height = reference.shape[1]
        grid_msg.info.origin.position.x = -reference.half_extent
        grid_msg.info.origin.position.y = -reference.half_extent
        grid_msg.info.origin.orientation.w = 1.0
        # OccupancyGrid is row-major in y; the belief array is indexed [x, y].
        grid_msg.data = scaled.T.flatten().tolist()
        self.grid_pub.publish(grid_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RFLocalizerNode()
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
