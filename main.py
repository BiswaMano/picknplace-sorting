"""
VLA-AutoParts — main entry point.

Subcommands:
  dataset   — dataset pipeline (download, clean, annotate, split)
  annotate  — Gradio annotation tool
  dream     — generate Sort Dreaming evaluation set
  train     — QLoRA fine-tuning
  eval      — evaluation suite
  demo      — Gradio demo app

Run:
  python main.py --help
  python main.py dataset build --rf-key YOUR_KEY
  python main.py train
  python main.py eval
  python main.py demo
"""

import typer

app = typer.Typer(
    name="vla-autoparts",
    help="VLA model for automotive parts recognition and sorting.",
    add_completion=False,
)


@app.command()
def dataset(
    action: str = typer.Argument(
        "build",
        help="Pipeline action: build | clean | annotate-auto | split | validate"
    ),
    rf_key: str = typer.Option("", help="Roboflow API key"),
):
    """Dataset pipeline commands."""
    from src.dataset.pipeline import app as pipeline_app
    import sys
    args = [action]
    if rf_key:
        args += ["--rf-key", rf_key]
    sys.argv = ["pipeline"] + args
    pipeline_app()


@app.command()
def annotate():
    """Launch Gradio annotation tool."""
    from src.dataset.annotator import main
    main()


@app.command()
def dream(
    count: int = typer.Option(500),
    seed: int = typer.Option(42),
):
    """Generate the Sort Dreaming evaluation set."""
    from src.dataset.sort_dreaming import generate_sort_dreaming
    from pathlib import Path
    import jsonlines

    out = Path("data/annotations/sort_dreaming.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_sort_dreaming(count=count, seed=seed)
    with jsonlines.open(out, mode="w") as w:
        w.write_all(samples)
    print(f"Generated {len(samples)} Sort Dreaming samples → {out}")


@app.command()
def train(
    epochs: int   = typer.Option(3),
    lr: float     = typer.Option(2e-4),
    lora_r: int   = typer.Option(16),
    batch_size: int = typer.Option(1),
    grad_accum: int = typer.Option(8),
):
    """Run QLoRA fine-tuning."""
    from src.training.trainer import VLATrainer, default_config
    from pathlib import Path

    cfg = default_config()
    cfg.update({
        "epochs":           epochs,
        "lr":               lr,
        "lora_r":           lora_r,
        "batch_size":       batch_size,
        "grad_accum_steps": grad_accum,
    })
    trainer = VLATrainer(cfg, out_dir=Path("checkpoints"))
    trainer.train()


@app.command()
def eval(
    checkpoint: str = typer.Option("", help="Path to checkpoint (empty = current weights)"),
    sort_dreaming: bool = typer.Option(True),
    baseline: bool = typer.Option(False),
):
    """Run evaluation suite."""
    from src.eval.benchmark import VLAEvaluator, _print_results
    from pathlib import Path

    ckpt = Path(checkpoint) if checkpoint else None
    evaluator = VLAEvaluator(checkpoint_path=ckpt)
    evaluator.load_model()

    results = evaluator.eval_test_set()
    _print_results("Test Set", results)

    if sort_dreaming:
        sd_results = evaluator.eval_sort_dreaming()
        _print_results("Sort Dreaming", sd_results)

    if baseline:
        bl_results = evaluator.eval_baseline_zero_shot()
        _print_results("Zero-Shot Baseline", bl_results)


@app.command()
def simulate(
    part: str = typer.Option("brake_caliper", help="Part slug"),
    condition: str = typer.Option("new", help="Condition label"),
    sample_id: str = typer.Option("demo_0001", help="Sample ID for pick position seed"),
    plot: bool = typer.Option(True, help="Generate trajectory plots"),
    view: str = typer.Option("both", help="2d | 3d | both"),
):
    """Simulate a pick-and-place trajectory for a given part + condition."""
    from src.simulation.trajectory import generate_trajectory_for_sample
    from src.simulation.visualize import (
        plot_workspace_2d, plot_trajectory_2d,
        plot_trajectory_3d, save_figure,
    )
    from pathlib import Path

    traj = generate_trajectory_for_sample(part, condition, sample_id=sample_id)
    print(traj)

    if plot:
        out_dir = Path("results/trajectories")
        if view in ("2d", "both"):
            fig, ax = plot_workspace_2d()
            plot_trajectory_2d(traj, ax=ax, title=f"{part} ({condition}) → Bin {traj.target_bin}")
            save_figure(fig, out_dir / f"{part}_{condition}_2d.png")

        if view in ("3d", "both"):
            fig, ax = plot_trajectory_3d(traj, title=f"{part} ({condition}) → Bin {traj.target_bin}")
            save_figure(fig, out_dir / f"{part}_{condition}_3d.png")


@app.command()
def demo(
    checkpoint: str = typer.Option("", help="Checkpoint path"),
    port: int = typer.Option(7861),
    share: bool = typer.Option(False),
):
    """Launch the Gradio demo."""
    from demo.app import build_demo
    ckpt = checkpoint or None
    d = build_demo(checkpoint_path=ckpt)
    d.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    app()
