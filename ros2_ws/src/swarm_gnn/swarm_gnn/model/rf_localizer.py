"""RF emitter localisation: propagation model + recursive Bayesian estimator.

The mission turns on finding emitters the team cannot see directly. Each
platform's DF array yields two weak measurements === a received power and a
bearing, both noisy, both biased by the urban clutter === and no single
observation localises anything. What *does* localise an emitter is accumulating
many weak cuts from spatially separated platforms, which is precisely why this
is a swarm problem and why the allocator spends tasks on triangulation
baselines instead of just driving at the strongest signal.

Two estimators run in series:

  1. A log-probability occupancy grid over the sector, updated recursively
     (Bayes) with every detection. This is the robust, multi-hypothesis stage:
     it copes with multipath outliers and with several emitters at once, and it
     never needs an initial guess. Its posterior is what the mission executive
     reads to decide whether a cell is worth a closer look.

  2. Once the grid posterior concentrates, a Gauss-Newton refinement over the
     raw bearings (classic DF multilateration) sharpens the estimate and gives
     a covariance, which the executive thresholds to promote a candidate to
     LOCALIZED and open confirm/CBRN tasks.

Stage 1 alone is too coarse to drive a UGV to a 3 m acceptance radius; stage 2
alone diverges when the first few cuts are bad. Together they go from nothing to
a metre-level fix with the handful of observations a three-robot team can get.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ========================= propagation model ================================
# Log-distance path loss with an urban exponent. The simulated sensor and the
# estimator share this function on purpose: the estimator's model error is then
# exactly the shadowing and bearing noise, not a mismatched parameterisation,
# which keeps the study focused on the coordination question.

#: Path-loss exponent. 2.0 is free space; dense urban is 2.7-3.5.
PATH_LOSS_EXPONENT = 2.9
#: Reference distance (m) and the loss at it, for a 433 MHz dipole pair.
REF_DISTANCE = 1.0
REF_LOSS_DB = 26.0
#: Log-normal shadowing standard deviation, dB. Urban NLOS is 6-10 dB.
SHADOWING_SIGMA_DB = 6.5
#: Prior uncertainty on an unknown emitter's transmit power, dB. A hand-built
#: command trigger and a vehicle-mounted jammer differ by well over 15 dB, and
#: the team cannot know which it is hearing, so this is genuinely wide.
TX_POWER_SIGMA_DB = 9.0


def expected_rssi(
    tx_power_dbm: float,
    distance: float,
    exponent: float = PATH_LOSS_EXPONENT,
    extra_loss_db: float = 0.0,
) -> float:
    """Mean received power (dBm) at `distance` metres from the emitter."""
    d = max(distance, 0.35)
    loss = REF_LOSS_DB + 10.0 * exponent * math.log10(d / REF_DISTANCE)
    return tx_power_dbm - loss - extra_loss_db


def rssi_to_range(
    rssi_dbm: float,
    tx_power_dbm: float,
    exponent: float = PATH_LOSS_EXPONENT,
) -> float:
    """Invert the path-loss model. Only as good as the assumed tx power, which
    is why range is used as a weak prior and bearing carries the geometry."""
    loss = tx_power_dbm - rssi_dbm - REF_LOSS_DB
    return REF_DISTANCE * (10.0 ** (loss / (10.0 * exponent)))


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]. Bearing residuals are meaningless without this."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Detection:
    """One DF observation, mirroring swarm_interfaces/ThreatDetection."""

    observer_id: str
    position: tuple[float, float, float]
    rssi_dbm: float
    bearing_rad: float          # in the map frame; NaN when the cut is unusable
    bearing_sigma_rad: float
    frequency_mhz: float
    stamp: float = 0.0

    @property
    def has_bearing(self) -> bool:
        return not math.isnan(self.bearing_rad)


# ====================== stage 1: Bayesian occupancy grid =====================
class BayesianRFGrid:
    """Recursive log-odds grid over emitter position.

    State is log p(emitter at cell) up to a constant. Each detection
    contributes an additive log-likelihood field: a bearing fan (von Mises in
    angle) multiplied by a range shell (Gaussian in log-distance from the RSSI
    inversion). Because the update is additive in log space it is order-free and
    O(cells) per detection, so every robot can run its own copy and the copies
    converge to the same posterior once their detections have been gossiped.
    """

    def __init__(
        self,
        half_extent: float = 60.0,
        resolution: float = 1.5,
        tx_power_prior_dbm: float = 22.0,
        shadowing_sigma_db: float = SHADOWING_SIGMA_DB,
        tx_power_sigma_db: float = TX_POWER_SIGMA_DB,
    ) -> None:
        self.half_extent = float(half_extent)
        self.resolution = float(resolution)
        self.tx_power_prior_dbm = float(tx_power_prior_dbm)
        self.shadowing_sigma_db = float(shadowing_sigma_db)
        self.tx_power_sigma_db = float(tx_power_sigma_db)
        # Transmit power is a nuisance parameter, not a known constant: the
        # team has no way to measure what an emitter radiates. Marginalising a
        # Gaussian prior over it out of the measurement model simply adds its
        # variance to the shadowing variance, so the range term is governed by
        # this combined sigma.
        #
        # This matters more than it looks. With tx power treated as known, a
        # wrong prior makes the implied range wrong, the posterior piles up
        # against whichever grid edge is closest to that wrong range, and the
        # filter reports a *confident* fix in the sector corner === observed in
        # a live run as a 0.8 m sigma track at (59.2, 59.2). Widening the term
        # to its true uncertainty makes RSSI a weak range cue, which is all it
        # honestly is, and lets the bearing intersection do the localising.
        self.range_sigma_db = math.sqrt(
            self.shadowing_sigma_db**2 + self.tx_power_sigma_db**2
        )

        n = int(round(2 * half_extent / resolution))
        self.shape = (n, n)
        # Cell centres in the map frame.
        axis = -half_extent + resolution * (np.arange(n) + 0.5)
        self.xs, self.ys = np.meshgrid(axis, axis, indexing="ij")
        # Uniform prior in log space.
        self.log_belief = np.zeros(self.shape, dtype=np.float64)
        self.num_detections = 0

    # ----------------------------------------------------------------- update
    def update(self, det: Detection) -> None:
        """Fold one detection into the posterior."""
        dx = self.xs - det.position[0]
        dy = self.ys - det.position[1]
        # Include the observer's height: a drone at 14 m is genuinely further
        # from a ground emitter than its map-frame footprint suggests.
        dz = det.position[2]
        planar = np.hypot(dx, dy)
        slant = np.sqrt(planar**2 + dz**2)

        log_lik = np.zeros(self.shape, dtype=np.float64)

        # --- range likelihood from received power -------------------------
        # Gaussian in dB space: p(rssi | cell) with the shadowing sigma.
        predicted = self.tx_power_prior_dbm - (
            REF_LOSS_DB + 10.0 * PATH_LOSS_EXPONENT * np.log10(np.maximum(slant, 0.35))
        )
        resid_db = det.rssi_dbm - predicted
        log_lik += -0.5 * (resid_db / self.range_sigma_db) ** 2

        # --- bearing likelihood -------------------------------------------
        if det.has_bearing:
            cell_bearing = np.arctan2(dy, dx)
            ang_resid = np.arctan2(
                np.sin(cell_bearing - det.bearing_rad),
                np.cos(cell_bearing - det.bearing_rad),
            )
            sigma = max(det.bearing_sigma_rad, 1e-3)
            # Von Mises with concentration 1/sigma^2, which for small sigma is
            # the wrapped Gaussian; written in log form for stability.
            kappa = 1.0 / (sigma**2)
            bearing_ll = kappa * (np.cos(ang_resid) - 1.0)

            # A cell almost on top of the observer subtends every bearing at
            # once, so the measured bearing says nothing about it: the right
            # treatment is a *neutral* bearing likelihood there, not a
            # penalised one.
            #
            # Crucially this neutralises only the bearing term. Clamping the
            # combined likelihood instead would also erase the range penalty,
            # which makes the observer's own position the most probable
            # emitter site on every single detection and accumulates into a
            # confident ghost emitter underneath the robot. That bug was
            # observed in a live run before this was split out.
            near = planar < 2.0 * self.resolution
            bearing_ll[near] = 0.0
            log_lik += bearing_ll

        self.log_belief += log_lik
        # Renormalise to keep the exponential in `posterior` in range.
        self.log_belief -= self.log_belief.max()
        self.num_detections += 1

    def decay(self, factor: float = 0.985) -> None:
        """Shrink the log-odds toward the uniform prior.

        Emitters are mobile and can be switched off, so an old posterior must
        not pin the team to a stale cell. Without this the grid is a monument to
        the first few detections.
        """
        self.log_belief *= factor

    # ------------------------------------------------------------- read-out
    def posterior(self) -> np.ndarray:
        """Normalised probability over cells."""
        p = np.exp(self.log_belief - self.log_belief.max())
        total = p.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.full(self.shape, 1.0 / p.size)
        return p / total

    def peak(self) -> tuple[tuple[float, float], float]:
        """((x, y), posterior mass of the peak cell)."""
        p = self.posterior()
        idx = int(np.argmax(p))
        i, j = np.unravel_index(idx, self.shape)
        return (float(self.xs[i, j]), float(self.ys[i, j])), float(p[i, j])

    def modes(self, min_separation: float = 12.0, max_modes: int = 6,
              mass_threshold: float = 0.004,
              edge_margin: float = 6.0) -> list[tuple[tuple[float, float], float]]:
        """Non-maximum-suppressed local maxima, excluding the domain boundary.

        Multiple emitters are the normal case, so the read-out has to be
        multi-modal; taking only the global peak would hide every emitter but
        the loudest. Returns [((x, y), local mass), ...], strongest first.

        `edge_margin` discards peaks that close to the grid boundary. A peak
        pinned to the edge is not evidence of an emitter there, it is evidence
        that the posterior mass wanted to leave the domain and was truncated,
        which is what happens when the assumed transmit power implies a range
        beyond the sector. A live run produced confidently-localised tracks at
        (11.2, 59.2) and (0.8, 59.2) on a grid bounded at 60 m, and the team
        then spent real tasks investigating them. Emitters are known to lie
        inside the sector, so the boundary band holds no usable hypothesis.
        """
        p = self.posterior().copy()
        out: list[tuple[tuple[float, float], float]] = []
        radius_cells = max(1, int(round(min_separation / self.resolution)))

        # Zero the boundary band outright, so suppression cannot walk into it.
        if edge_margin > 0.0:
            band = max(1, int(round(edge_margin / self.resolution)))
            p[:band, :] = 0.0
            p[-band:, :] = 0.0
            p[:, :band] = 0.0
            p[:, -band:] = 0.0

        for _ in range(max_modes):
            idx = int(np.argmax(p))
            i, j = np.unravel_index(idx, self.shape)
            if p[i, j] < mass_threshold:
                break
            # Local mass is the suppressed window's total, which is a better
            # confidence measure than a single cell's probability.
            lo_i, hi_i = max(0, i - radius_cells), min(self.shape[0], i + radius_cells + 1)
            lo_j, hi_j = max(0, j - radius_cells), min(self.shape[1], j + radius_cells + 1)
            window_mass = float(p[lo_i:hi_i, lo_j:hi_j].sum())
            out.append(((float(self.xs[i, j]), float(self.ys[i, j])), window_mass))
            p[lo_i:hi_i, lo_j:hi_j] = 0.0
        return out

    def covariance_at(self, centre: tuple[float, float], window: float = 10.0) -> np.ndarray:
        """2x2 position covariance of the posterior inside a window.

        This is what the mission executive thresholds: a wide covariance means
        the cut geometry is poor and another triangulation baseline is worth
        more than driving in.
        """
        p = self.posterior()
        sel = (np.abs(self.xs - centre[0]) <= window) & (np.abs(self.ys - centre[1]) <= window)
        mass = p[sel].sum()
        if mass <= 1e-12:
            return np.eye(2) * (self.half_extent**2)
        w = p[sel] / mass
        dx = self.xs[sel] - centre[0]
        dy = self.ys[sel] - centre[1]
        cxx = float((w * dx * dx).sum())
        cyy = float((w * dy * dy).sum())
        cxy = float((w * dx * dy).sum())
        # Floor at the discretisation limit: the grid cannot resolve better.
        floor = (self.resolution / 2.0) ** 2
        return np.array([[max(cxx, floor), cxy], [cxy, max(cyy, floor)]])


# =================== stage 2: Gauss-Newton multilateration ==================
def observer_baseline(detections: list[Detection]) -> float:
    """Largest separation between any two observers that supplied a bearing."""
    pts = np.array([d.position[:2] for d in detections if d.has_bearing], dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    best = 0.0
    for i in range(len(pts)):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        if d.size:
            best = max(best, float(d.max()))
    return best


def multilaterate(
    detections: list[Detection],
    initial: tuple[float, float],
    iterations: int = 24,
    huber_delta: float = 2.5,
    min_baseline: float = 8.0,
) -> tuple[tuple[float, float], np.ndarray, float] | None:
    """Refine an emitter fix from bearing cuts. Needs >= 2 usable bearings.

    Iteratively reweighted Gauss-Newton on bearing residuals, with a Huber
    weight so one multipath cut cannot drag the solution. Returns
    ((x, y), 2x2 covariance, rms residual in radians), or None when the fix
    cannot be trusted.

    On the returned covariance
    -------------------------
    The covariance is the inverse Fisher information of the *bearing noise*
    alone. It does not know about geometric dilution: with every observer
    bunched together, the along-range direction is unobservable, yet the
    inverse-information matrix still reports a small sigma. Measured here, a
    1.4 m baseline produced a 60 m position error reported with a 2.9 m sigma.

    That trap is closed by `min_baseline`: a cut set whose observers span less
    than that distance is rejected outright rather than refined, because no
    amount of repeated measurement from one spot recovers range. This is the
    whole reason the mission plan spends RF_TRIANGULATE tasks on *spreading the
    observers* instead of just accumulating more samples.
    """
    cuts = [d for d in detections if d.has_bearing]
    if len(cuts) < 2:
        return None
    if observer_baseline(cuts) < min_baseline:
        return None

    x = np.array([initial[0], initial[1]], dtype=np.float64)

    for _ in range(iterations):
        J = np.zeros((len(cuts), 2), dtype=np.float64)
        r = np.zeros(len(cuts), dtype=np.float64)
        w = np.zeros(len(cuts), dtype=np.float64)

        for k, d in enumerate(cuts):
            dx = x[0] - d.position[0]
            dy = x[1] - d.position[1]
            d2 = dx * dx + dy * dy
            if d2 < 1e-6:
                return None
            predicted = math.atan2(dy, dx)
            resid = wrap_angle(d.bearing_rad - predicted)
            # d(atan2)/dx = -dy/d2, d/dy = dx/d2; residual is (meas - pred) so
            # the Jacobian of the residual carries the opposite sign.
            J[k, 0] = dy / d2
            J[k, 1] = -dx / d2
            r[k] = resid
            sigma = max(d.bearing_sigma_rad, 1e-3)
            # Inverse-variance weight, Huber-downweighted on large residuals.
            scaled = abs(resid) / sigma
            huber = 1.0 if scaled <= huber_delta else huber_delta / scaled
            w[k] = huber / (sigma**2)

        W = np.diag(w)
        H = J.T @ W @ J
        # Reject degenerate geometry rather than returning a confident wrong fix.
        if np.linalg.cond(H) > 1e8:
            return None
        try:
            # Gauss-Newton normal equations: (J^T W J) step = -J^T W r.
            step = -np.linalg.solve(H, J.T @ W @ r)
        except np.linalg.LinAlgError:
            return None

        # Trust region: a DF update should never teleport the estimate.
        norm = float(np.linalg.norm(step))
        if norm > 15.0:
            step *= 15.0 / norm
            norm = 15.0
        x = x + step
        if norm < 1e-4:
            break

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt(np.mean(r**2)))
    if not np.isfinite(cov).all() or not np.isfinite(x).all():
        return None
    return (float(x[0]), float(x[1])), cov, rms


@dataclass
class ThreatTrack:
    """A fused emitter hypothesis maintained across the mission.

    `confirm_count` is the number of *independent sensor modalities* that have
    corroborated it. Two are required before a marker is authorised, which is
    the mission rule that protects the civilian transmitter.

    `evidence` holds the raw observations those modalities produced, and the
    class is *derived* from them rather than asserted by whichever robot
    happened to look last. That matters because no single sensor separates the
    three emitter types:

      thermal_hot   a powered emitter is really here (any of the three)
      cbrn_pos      explosive precursor present  -> IED trigger
      cbrn_neg      clean sample                 -> not a trigger
      jam_detected  comms degraded at close range -> active jammer
      no_jam        comms clean at close range    -> not a jammer

    A jammer and a licensed civilian transmitter are indistinguishable on RF
    power and both read cold on CBRN; only the jamming measurement separates
    them. An earlier version had each robot assert a class directly, so a
    clean CBRN sample reported "civilian" === and the jammer, which has no
    chemical signature, was classified no-strike and could never be
    neutralised. That single line was why the mission produced no results.
    """

    threat_id: str
    position: tuple[float, float]
    #: Frequency channel this track belongs to. A channel carries one
    #: transmitter, so it is also the strongest association cue available.
    channel: int = 0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 400.0)
    confidence: float = 0.0
    frequency_mhz: float = 433.0
    emitter_class: int = 0           # ThreatEstimate CLASS_*
    state: int = 0                   # ThreatEstimate STATE_*
    confirm_count: int = 0
    detections: list[Detection] = field(default_factory=list)
    #: Raw observation tokens; see the class docstring.
    evidence: set[str] = field(default_factory=set)
    #: Distinct sensor modalities that have reported, e.g. {"thermal", "cbrn"}.
    modalities: set[str] = field(default_factory=set)
    last_update: float = 0.0
    #: Observer positions already used for a bearing, so the executive can tell
    #: whether another triangulation baseline would actually add geometry.
    baseline_positions: list[tuple[float, float]] = field(default_factory=list)

    @property
    def position_sigma(self) -> float:
        """Scalar 1-sigma radius, the larger principal axis."""
        try:
            eig = np.linalg.eigvalsh(self.covariance)
            return float(math.sqrt(max(eig.max(), 0.0)))
        except np.linalg.LinAlgError:
            return float("inf")

    def classify(self) -> int:
        """Derive the emitter class from accumulated evidence.

        Returns a ThreatDetection CLASS_* value. Ordering matters: a positive
        chemical sample is decisive, an active jamming measurement is decisive
        in its absence, and "civilian" is only concluded once both discriminators
        have actually come back negative === never by default.
        """
        if "cbrn_pos" in self.evidence:
            return 2       # CLASS_IED_TRIGGER
        if "jam_detected" in self.evidence:
            return 1       # CLASS_JAMMER
        if "thermal_hot" in self.evidence and {"cbrn_neg", "no_jam"} & self.evidence:
            return 3       # CLASS_CIVILIAN === no-strike
        return 0           # CLASS_UNKNOWN

    def baseline_spread(self) -> float:
        """Largest separation between observers that contributed a bearing.

        A near-zero spread means every cut came from nearly the same place, so
        the fix is along-range-ambiguous no matter how many detections there are.
        """
        if len(self.baseline_positions) < 2:
            return 0.0
        pts = np.asarray(self.baseline_positions)
        best = 0.0
        for i in range(len(pts)):
            d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            if d.size:
                best = max(best, float(d.max()))
        return best
