"""
Simulated workspace for VLA-AutoParts pick-and-place.

Defines the physical layout:
  - Conveyor belt (pickup zone) on the left
  - 10 sorting bins arranged in two columns on the right
  - Robot arm operating envelope

All coordinates are in meters.

  Y
  0.9 ┌──────────┐      ┌───┐  ┌───┐
      │ Conveyor │      │ 1 │  │ 6 │
  0.7 │ (pickup  │      ├───┤  ├───┤
      │  zone)   │      │ 2 │  │ 7 │
  0.5 │          │      ├───┤  ├───┤
      │          │      │ 3 │  │ 8 │
  0.3 │          │      ├───┤  ├───┤
      │          │      │ 4 │  │ 9 │
  0.1 └──────────┘      ├───┤  ├───┤
                        │ 5 │  │10 │
  0.0                   └───┘  └───┘
      0.0  0.1  0.3     0.65  0.85  1.0  → X
"""

from dataclasses import dataclass, field
import random


@dataclass
class WorkspaceConfig:
    # Conveyor belt bounds (pickup zone)
    conveyor_x_min: float = 0.10
    conveyor_x_max: float = 0.35
    conveyor_y_min: float = 0.15
    conveyor_y_max: float = 0.85
    conveyor_z: float = 0.02           # table surface height

    # Bin positions: dict[bin_id -> (x, y, z center)]
    bin_positions: dict = field(default_factory=dict)

    # Robot arm parameters
    safe_z: float = 0.30               # clearance height for horizontal moves
    approach_z_offset: float = 0.08    # hover above pick/place point
    retreat_z_offset: float = 0.10     # retreat height above place point

    # Gripper
    grip_depth: float = 0.01           # how far below surface to grip

    def __post_init__(self):
        if not self.bin_positions:
            self.bin_positions = self._default_bin_positions()

    @staticmethod
    def _default_bin_positions() -> dict:
        """
        10 bins in two columns of 5.

        Column 1 (x=0.65): bins 1-5   (braking + suspension)
        Column 2 (x=0.85): bins 6-10  (engine + electrical + reject)
        """
        positions = {}
        y_values = [0.82, 0.66, 0.50, 0.34, 0.18]  # top to bottom

        # Column 1: bins 1-5
        for i, y in enumerate(y_values):
            positions[i + 1] = (0.65, y, 0.05)

        # Column 2: bins 6-10
        for i, y in enumerate(y_values):
            positions[i + 6] = (0.85, y, 0.05)

        return positions


# Singleton default workspace
DEFAULT_WORKSPACE = WorkspaceConfig()


def random_pick_point(
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """
    Sample a random pick position on the conveyor belt.

    Parts land at slightly different spots — this adds realistic
    variation to the training data.
    """
    r = rng or random.Random()
    x = r.uniform(config.conveyor_x_min, config.conveyor_x_max)
    y = r.uniform(config.conveyor_y_min, config.conveyor_y_max)
    z = config.conveyor_z
    return (round(x, 4), round(y, 4), round(z, 4))


def get_bin_position(
    bin_id: int,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
) -> tuple[float, float, float]:
    """Get the center position of a bin."""
    if bin_id not in config.bin_positions:
        raise ValueError(f"Invalid bin_id {bin_id}. Must be 1-10.")
    return config.bin_positions[bin_id]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DEFAULT_WORKSPACE
    print("Conveyor zone:")
    print(f"  X: [{cfg.conveyor_x_min}, {cfg.conveyor_x_max}]")
    print(f"  Y: [{cfg.conveyor_y_min}, {cfg.conveyor_y_max}]")
    print(f"  Z: {cfg.conveyor_z}")
    print()
    print("Bin positions:")
    for bid, pos in sorted(cfg.bin_positions.items()):
        print(f"  Bin {bid:2d}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
    print()
    print("Random pick points:")
    rng = random.Random(42)
    for _ in range(5):
        pt = random_pick_point(cfg, rng)
        print(f"  ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
