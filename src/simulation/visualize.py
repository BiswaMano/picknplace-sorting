"""
Trajectory visualization for VLA-AutoParts.

Provides 2D top-down and 3D views of the workspace and trajectories.

Usage:
  python -m src.simulation.visualize
  python -m src.simulation.visualize --part brake_caliper --condition severely_damaged
"""

import typer
import sys
import site
from pathlib import Path

# ── Fix system/pip mpl_toolkits conflict ──────────────────────────────────
# Must run BEFORE importing matplotlib, which tries to import mpl_toolkits
# internally. The system mpl_toolkits (compiled against matplotlib 3.5)
# is incompatible with pip matplotlib (3.9+).
_user_site = site.getusersitepackages()
_pip_mpl_path = str(Path(_user_site) / "mpl_toolkits")
if Path(_pip_mpl_path).exists():
    import mpl_toolkits
    mpl_toolkits.__path__ = [_pip_mpl_path]

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 3D support — register the projection after fixing the path
HAS_3D = False
try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.projections import projection_registry
    try:
        projection_registry.get_projection_class('3d')
    except ValueError:
        projection_registry.register(Axes3D)
    HAS_3D = True
except (ImportError, Exception) as e:
    pass

from src.simulation.workspace import WorkspaceConfig, DEFAULT_WORKSPACE
from src.simulation.trajectory import (
    Trajectory,
    WAYPOINT_NAMES,
    generate_trajectory_for_sample,
)

app = typer.Typer()

# Bin category colors
BIN_COLORS = {
    1: "#e74c3c", 2: "#e74c3c", 3: "#e74c3c",   # braking = red
    4: "#3498db", 5: "#3498db",                    # suspension = blue
    6: "#f39c12", 7: "#f39c12",                    # engine = orange
    8: "#2ecc71", 9: "#2ecc71",                    # electrical = green
    10: "#7f8c8d",                                  # reject = gray
}

BIN_LABELS = {
    1: "B1\nBrake\nPrem.", 2: "B2\nBrake\nStd.", 3: "B3\nBrake\nLine",
    4: "B4\nSusp.\nSafety", 5: "B5\nSusp.\nRide",
    6: "B6\nEng.\nSensor", 7: "B7\nEng.\nMech.",
    8: "B8\nElec.\nClimate", 9: "B9\nAccess.\nMotor",
    10: "B10\nREJECT",
}


# ---------------------------------------------------------------------------
# 2D workspace plot
# ---------------------------------------------------------------------------

def plot_workspace_2d(
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    ax=None,
    figsize=(10, 8),
):
    """Draw the workspace: conveyor + bins, top-down view."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Conveyor belt
    conv = mpatches.FancyBboxPatch(
        (config.conveyor_x_min, config.conveyor_y_min),
        config.conveyor_x_max - config.conveyor_x_min,
        config.conveyor_y_max - config.conveyor_y_min,
        boxstyle="round,pad=0.02",
        facecolor="#ecf0f1",
        edgecolor="#95a5a6",
        linewidth=2,
    )
    ax.add_patch(conv)
    cx = (config.conveyor_x_min + config.conveyor_x_max) / 2
    cy = (config.conveyor_y_min + config.conveyor_y_max) / 2
    ax.text(cx, cy, "CONVEYOR\n(pickup zone)", ha="center", va="center",
            fontsize=10, color="#7f8c8d", style="italic")

    # Bins
    bin_size = 0.10
    for bid, (bx, by, bz) in config.bin_positions.items():
        rect = mpatches.FancyBboxPatch(
            (bx - bin_size / 2, by - bin_size / 2),
            bin_size, bin_size,
            boxstyle="round,pad=0.01",
            facecolor=BIN_COLORS.get(bid, "#bdc3c7"),
            edgecolor="white",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(bx, by, BIN_LABELS.get(bid, f"B{bid}"),
                ha="center", va="center", fontsize=6, color="white", weight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.00)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    return fig, ax


# ---------------------------------------------------------------------------
# 2D trajectory overlay
# ---------------------------------------------------------------------------

def plot_trajectory_2d(
    trajectory: Trajectory,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    ax=None,
    color="#2c3e50",
    highlight_bin: bool = True,
    title: str = "",
):
    """Plot a trajectory on the 2D workspace."""
    if ax is None:
        fig, ax = plot_workspace_2d(config)
    else:
        fig = ax.figure

    xs = [wp[0] for wp in trajectory.waypoints]
    ys = [wp[1] for wp in trajectory.waypoints]

    # Draw path
    ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.7, zorder=5)

    # Draw waypoints
    markers = ['v', 's', '^', 'D', 'v', 'o', '^']  # different shapes per phase
    colors_wp = ['#27ae60', '#27ae60', '#2980b9', '#8e44ad', '#e67e22', '#e74c3c', '#95a5a6']

    for i, (x, y, name) in enumerate(zip(xs, ys, WAYPOINT_NAMES)):
        ax.scatter(x, y, marker=markers[i], c=colors_wp[i], s=80, zorder=10,
                   edgecolors='white', linewidths=1)
        offset_y = 0.03 if i % 2 == 0 else -0.04
        ax.annotate(f"{i+1}.{name}", (x, y), textcoords="offset points",
                    xytext=(8, 8 if i < 3 else -12), fontsize=7, color=color)

    # Highlight target bin
    if highlight_bin:
        bx, by, _ = config.bin_positions[trajectory.target_bin]
        ax.scatter(bx, by, marker='*', c='yellow', s=200, zorder=15,
                   edgecolors='red', linewidths=1.5)

    # Pick point marker
    px, py, _ = trajectory.pick_point
    ax.scatter(px, py, marker='x', c='red', s=100, zorder=15, linewidths=2)
    ax.annotate("PICK", (px, py), textcoords="offset points",
                xytext=(-20, -15), fontsize=8, color='red', weight='bold')

    if title:
        ax.set_title(title, fontsize=12, weight='bold')

    return fig, ax


# ---------------------------------------------------------------------------
# Side-view fallback (when 3D is not available)
# ---------------------------------------------------------------------------

def _plot_side_view(
    trajectory: Trajectory,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    figsize=(12, 6),
    title: str = "",
):
    """2D side view (X vs Z) showing the height profile of the trajectory."""
    fig, ax = plt.subplots(figsize=figsize)

    # Compute horizontal distance traveled (cumulative)
    wps = trajectory.waypoints
    horiz_dist = [0.0]
    for i in range(1, len(wps)):
        dx = wps[i][0] - wps[i-1][0]
        dy = wps[i][1] - wps[i-1][1]
        horiz_dist.append(horiz_dist[-1] + np.sqrt(dx**2 + dy**2))

    zs = [wp[2] for wp in wps]

    ax.plot(horiz_dist, zs, '-o', color='#2c3e50', linewidth=2, markersize=8)
    ax.fill_between(horiz_dist, 0, zs, alpha=0.1, color='#3498db')

    colors_wp = ['#27ae60', '#27ae60', '#2980b9', '#8e44ad', '#e67e22', '#e74c3c', '#95a5a6']
    for i, (hd, z, name) in enumerate(zip(horiz_dist, zs, WAYPOINT_NAMES)):
        ax.scatter(hd, z, c=colors_wp[i], s=80, zorder=10, edgecolors='white')
        ax.annotate(f"{i+1}.{name}", (hd, z), textcoords="offset points",
                    xytext=(5, 10), fontsize=8)

    # Draw table surface
    ax.axhline(y=config.conveyor_z, color='#95a5a6', linestyle='--', alpha=0.5, label='Table surface')
    ax.axhline(y=config.safe_z, color='#3498db', linestyle=':', alpha=0.5, label='Safe height')

    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_title(title or "Pick-and-Place Height Profile", fontsize=12, weight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 0.50)
    ax.grid(True, alpha=0.2)

    return fig, ax


# ---------------------------------------------------------------------------
# 3D trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_3d(
    trajectory: Trajectory,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    figsize=(12, 8),
    title: str = "",
):
    """3D plot showing the full trajectory including height."""
    if not HAS_3D:
        print("[warn] 3D plotting not available, generating 2D side-view instead")
        return _plot_side_view(trajectory, config, figsize, title)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xs = [wp[0] for wp in trajectory.waypoints]
    ys = [wp[1] for wp in trajectory.waypoints]
    zs = [wp[2] for wp in trajectory.waypoints]

    # Path
    ax.plot(xs, ys, zs, '-o', color='#2c3e50', linewidth=2, markersize=6)

    # Waypoint labels
    colors_wp = ['#27ae60', '#27ae60', '#2980b9', '#8e44ad', '#e67e22', '#e74c3c', '#95a5a6']
    for i, (x, y, z, name) in enumerate(zip(xs, ys, zs, WAYPOINT_NAMES)):
        ax.scatter(x, y, z, c=colors_wp[i], s=60, zorder=10)
        ax.text(x, y, z + 0.02, f"{i+1}.{name}", fontsize=7)

    # Draw conveyor surface
    cx = [config.conveyor_x_min, config.conveyor_x_max, config.conveyor_x_max, config.conveyor_x_min]
    cy = [config.conveyor_y_min, config.conveyor_y_min, config.conveyor_y_max, config.conveyor_y_max]
    cz = [config.conveyor_z] * 4
    ax.plot_trisurf(
        cx + [cx[0]], cy + [cy[0]], cz + [cz[0]],
        alpha=0.15, color='#ecf0f1'
    )

    # Draw bin positions
    for bid, (bx, by, bz) in config.bin_positions.items():
        ax.scatter(bx, by, bz, c=BIN_COLORS.get(bid, 'gray'), s=80, marker='s', alpha=0.6)
        ax.text(bx, by, bz + 0.02, f"B{bid}", fontsize=7, ha='center')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "Pick-and-Place Trajectory (3D)", fontsize=12, weight='bold')

    return fig, ax


# ---------------------------------------------------------------------------
# Comparison plot (predicted vs ground truth)
# ---------------------------------------------------------------------------

def plot_comparison(
    pred_trajectory: Trajectory,
    gt_trajectory: Trajectory,
    config: WorkspaceConfig = DEFAULT_WORKSPACE,
    figsize=(12, 8),
    title: str = "Predicted vs Ground Truth",
):
    """Overlay predicted and ground truth trajectories."""
    fig, ax = plot_workspace_2d(config, figsize=figsize)

    # Ground truth in green
    gt_xs = [wp[0] for wp in gt_trajectory.waypoints]
    gt_ys = [wp[1] for wp in gt_trajectory.waypoints]
    ax.plot(gt_xs, gt_ys, '--', color='#27ae60', linewidth=2, alpha=0.8, label='Ground Truth', zorder=5)
    ax.scatter(gt_xs, gt_ys, c='#27ae60', s=50, zorder=10, edgecolors='white')

    # Predicted in red
    pred_xs = [wp[0] for wp in pred_trajectory.waypoints]
    pred_ys = [wp[1] for wp in pred_trajectory.waypoints]
    ax.plot(pred_xs, pred_ys, '-', color='#e74c3c', linewidth=2, alpha=0.8, label='Predicted', zorder=6)
    ax.scatter(pred_xs, pred_ys, c='#e74c3c', s=50, zorder=11, edgecolors='white')

    # Error lines between corresponding waypoints
    for i in range(len(gt_xs)):
        ax.plot([gt_xs[i], pred_xs[i]], [gt_ys[i], pred_ys[i]],
                ':', color='gray', linewidth=1, alpha=0.5)
        err = np.sqrt((gt_xs[i] - pred_xs[i])**2 + (gt_ys[i] - pred_ys[i])**2)
        if err > 0.01:
            mid_x = (gt_xs[i] + pred_xs[i]) / 2
            mid_y = (gt_ys[i] + pred_ys[i]) / 2
            ax.text(mid_x, mid_y, f"{err:.3f}m", fontsize=6, color='gray')

    ax.legend(loc='upper left')
    ax.set_title(title, fontsize=12, weight='bold')
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def visualize(
    part: str = typer.Option("brake_caliper", help="Part slug"),
    condition: str = typer.Option("new", help="Condition"),
    sample_id: str = typer.Option("demo_0001", help="Sample ID (affects pick position)"),
    view: str = typer.Option("both", help="2d | 3d | both"),
    out_dir: Path = typer.Option(Path("results/trajectories"), help="Output directory"),
):
    """Generate and visualize a trajectory."""
    traj = generate_trajectory_for_sample(part, condition, sample_id=sample_id)
    print(traj)

    if view in ("2d", "both"):
        fig, ax = plot_workspace_2d()
        plot_trajectory_2d(traj, ax=ax, title=f"{part} ({condition}) → Bin {traj.target_bin}")
        save_figure(fig, out_dir / f"{part}_{condition}_2d.png")

    if view in ("3d", "both"):
        fig, ax = plot_trajectory_3d(traj, title=f"{part} ({condition}) → Bin {traj.target_bin}")
        save_figure(fig, out_dir / f"{part}_{condition}_3d.png")


if __name__ == "__main__":
    app()
