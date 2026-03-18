"""
Deterministic trajectory generator for VLA-AutoParts pick-and-place.

Given a pick point on the conveyor and a target bin, generates a 7-waypoint
trajectory that a robot arm would follow:

  1. approach  — hover above the part
  2. pick      — descend and grasp
  3. lift      — lift straight up to safe height
  4. move      — horizontal transit to above the target bin
  5. lower     — descend above the bin
  6. place     — lower into bin and release
  7. retreat   — lift away from the bin

This module is the trajectory equivalent of sorting_rules.py —
it generates the GROUND TRUTH trajectories for training data.
The model then learns to predict these trajectories.

Usage:
  python -m src.simulation.trajectory
"""

import hashlib
import random
from dataclasses import dataclass, field

from src.simulation.workspace import (
    WorkspaceConfig,
    DEFAULT_WORKSPACE,
    random_pick_point,
    get_bin_position,
)
from src.dataset.sorting_rules import sort_by_slug


NUM_WAYPOINTS = 7
WAYPOINT_NAMES = [
    "approach",   # 0: hover above part
    "pick",       # 1: descend to grasp
    "lift",       # 2: lift to safe height
    "move",       # 3: horizontal transit at safe height
    "lower",      # 4: descend above bin
    "place",      # 5: release into bin
    "retreat",    # 6: lift away
]


@dataclass
class Trajectory:
    """A 7-waypoint pick-and-place trajectory."""

    waypoints: list[tuple[float, float, float]]   # 7 x (x, y, z)
    waypoint_names: list[str] = field(default_factory=lambda: list(WAYPOINT_NAMES))
    pick_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_bin: int = 1

    def __post_init__(self):
        assert len(self.waypoints) == NUM_WAYPOINTS, (
            f"Expected {NUM_WAYPOINTS} waypoints, got {len(self.waypoints)}"
        )

    def flat(self) -> list[float]:
        """Flatten to a list of 21 floats (7 waypoints x 3 coords)."""
        return [c for wp in self.waypoints for c in wp]

    @classmethod
    def from_flat(
        cls,
        values: list[float],
        pick_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_bin: int = 1,
    ) -> "Trajectory":
        """Reconstruct from a flat list of 21 floats."""
        assert len(values) == NUM_WAYPOINTS * 3
        waypoints = [
            (values[i], values[i + 1], values[i + 2])
            for i in range(0, len(values), 3)
        ]
        return cls(waypoints=waypoints, pick_point=pick_point, target_bin=target_bin)

    def __repr__(self) -> str:
        lines = [f"Trajectory(bin={self.target_bin}):"]
        for name, wp in zip(self.waypoint_names, self.waypoints):
            lines.append(f"  {name:10s}  x={wp[0]:.4f}  y={wp[1]:.4f}  z={wp[2]:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_trajectory(
    pick_point: tuple[float, float, float],
    target_bin: int,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
) -> Trajectory:
    """
    Generate a 7-waypoint pick-and-place trajectory.

    Args:
        pick_point:  (x, y, z) where the part sits on the conveyor
        target_bin:  which bin to place it in (1-10)
        config:      workspace geometry

    Returns:
        Trajectory with 7 waypoints
    """
    px, py, pz = pick_point
    bx, by, bz = get_bin_position(target_bin, config)

    waypoints = [
        # 0: approach — hover above the part
        (px, py, pz + config.approach_z_offset),

        # 1: pick — descend to grasp (slightly below surface for firm grip)
        (px, py, pz - config.grip_depth),

        # 2: lift — straight up to safe height
        (px, py, config.safe_z),

        # 3: move — horizontal transit at safe height to above the bin
        (bx, by, config.safe_z),

        # 4: lower — descend to just above the bin
        (bx, by, bz + config.approach_z_offset),

        # 5: place — lower into the bin and release
        (bx, by, bz),

        # 6: retreat — lift away
        (bx, by, config.safe_z + config.retreat_z_offset),
    ]

    # Round for cleanliness
    waypoints = [(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in waypoints]

    return Trajectory(
        waypoints=waypoints,
        pick_point=pick_point,
        target_bin=target_bin,
    )


# ---------------------------------------------------------------------------
# Convenience: generate from a sample record
# ---------------------------------------------------------------------------

def generate_trajectory_for_sample(
    part_slug: str,
    condition: str,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    sample_id: str = "",
) -> Trajectory:
    """
    Generate a trajectory for a dataset sample.

    Uses the sorting rules to determine the target bin, then generates
    a random pick point (seeded by sample_id for reproducibility) and
    the full trajectory.

    Args:
        part_slug:  e.g. "brake_caliper"
        condition:  e.g. "minor_corrosion"
        config:     workspace geometry
        sample_id:  used as seed for reproducible random pick positions

    Returns:
        Trajectory
    """
    # Determine target bin from sorting rules
    action = sort_by_slug(part_slug, condition)
    target_bin = action.bin

    # Create a reproducible RNG from the sample ID
    seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Random pick position on conveyor
    pick_point = random_pick_point(config, rng)

    return generate_trajectory(pick_point, target_bin, config)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trajectory for: brake_caliper + new ===")
    traj = generate_trajectory_for_sample("brake_caliper", "new", sample_id="demo_0001")
    print(traj)
    print(f"\nFlat: {traj.flat()}")
    print()

    print("=== Trajectory for: brake_caliper + severely_damaged ===")
    traj2 = generate_trajectory_for_sample("brake_caliper", "severely_damaged", sample_id="demo_0002")
    print(traj2)
    print()

    print("=== Trajectory for: oil_filter + good ===")
    traj3 = generate_trajectory_for_sample("oil_filter", "good", sample_id="demo_0003")
    print(traj3)
    print()

    # Show that different sample_ids produce different pick points
    print("=== Same part, different pick positions (different sample IDs) ===")
    for i in range(5):
        t = generate_trajectory_for_sample("spark_plug", "new", sample_id=f"spark_{i:04d}")
        print(f"  sample spark_{i:04d}: pick=({t.pick_point[0]:.3f}, {t.pick_point[1]:.3f}) → bin {t.target_bin}")
