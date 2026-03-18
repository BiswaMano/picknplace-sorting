"""
VLA-AutoParts dataset construction pipeline.

Steps:
  1. Download seed images from Roboflow datasets
  2. Clean and standardize (resize to 448×448, rename)
  3. Auto-generate action labels from sorting rules
  4. Format as InternVL2 conversation JSONL
  5. Stratified train/val/test split (70/15/15)

Usage:
  python -m src.dataset.pipeline --help
  python -m src.dataset.pipeline build --rf-key YOUR_KEY
  python -m src.dataset.pipeline format-only   # if images already collected
  python -m src.dataset.pipeline validate
"""

import json
import os
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import jsonlines
import typer
from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.model_selection import train_test_split

from src.dataset.classes import (
    PART_CLASSES,
    SLUG_TO_CLASS,
    CONDITIONS,
    NUM_CLASSES,
    NUM_BINS,
)
from src.dataset.sorting_rules import apply_sorting_rules, validate_annotation

app = typer.Typer(help="VLA-AutoParts dataset pipeline")
console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANN_DIR = DATA_DIR / "annotations"

IMAGE_SIZE = 448        # minimum side after resize
TARGET_PER_CLASS = 50  # images per class in final dataset

# ---------------------------------------------------------------------------
# Condition distribution targets (fractions)
# ---------------------------------------------------------------------------
CONDITION_DIST = {
    "new":               0.20,
    "good":              0.20,
    "minor_wear":        0.15,
    "minor_corrosion":   0.15,
    "damaged":           0.20,
    "severely_damaged":  0.10,
}

# ---------------------------------------------------------------------------
# Human prompt template (identical across all vision samples)
# ---------------------------------------------------------------------------
HUMAN_PROMPT = (
    "<image>\nIdentify this automotive part, assess its condition, "
    "and provide the sorting action."
)


# ---------------------------------------------------------------------------
# Step 1: Download from Roboflow
# ---------------------------------------------------------------------------
@app.command()
def download(
    rf_key: str = typer.Option(..., help="Roboflow API key"),
    workspace: str = typer.Option("team-data", help="Roboflow workspace"),
    project: str = typer.Option("car-parts-ybiev", help="Roboflow project slug"),
    version: int = typer.Option(1, help="Dataset version"),
    out_dir: Path = typer.Option(RAW_DIR, help="Output directory"),
):
    """Download a Roboflow dataset into data/raw/."""
    try:
        from roboflow import Roboflow
    except ImportError:
        console.print("[red]roboflow not installed. Run: pip install roboflow[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Downloading {workspace}/{project} v{version}...[/cyan]")
    rf = Roboflow(api_key=rf_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("folder", location=str(out_dir / project))
    console.print(f"[green]Downloaded to {dataset.location}[/green]")


# ---------------------------------------------------------------------------
# Step 2: Clean and standardize
# ---------------------------------------------------------------------------
def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)


def clean_image(src: Path, dst: Path, min_size: int = IMAGE_SIZE) -> bool:
    """
    Open an image, verify it, resize so the shorter side >= min_size,
    and save as JPEG.  Returns True on success.
    """
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            if min(w, h) < min_size:
                scale = min_size / min(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(dst, "JPEG", quality=92)
        return True
    except Exception as e:
        console.print(f"[yellow]Skipping {src.name}: {e}[/yellow]")
        return False


@app.command()
def clean(
    raw_dir: Path = typer.Option(RAW_DIR, help="Raw images root"),
    out_dir: Path = typer.Option(PROCESSED_DIR, help="Processed images root"),
):
    """
    Clean and standardize raw images.

    Expects raw_dir/<part_slug>/ subdirectories.
    Renames files to {slug}_{condition}_{id:04d}.jpg
    NOTE: condition is set to 'good' by default here; use the annotation
    tool (vla-annotate) to assign real conditions afterward.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_ok = 0

    for part in PART_CLASSES:
        src_class_dir = raw_dir / part.slug
        if not src_class_dir.exists():
            console.print(f"[yellow]Missing raw dir: {part.slug}[/yellow]")
            continue

        dst_class_dir = out_dir / part.slug
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in src_class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )

        ok = 0
        for i, img_path in enumerate(images):
            dst = dst_class_dir / f"{part.slug}_good_{i:04d}.jpg"
            if clean_image(img_path, dst):
                ok += 1

        console.print(f"  {part.slug}: {ok}/{len(images)} images processed")
        total_ok += ok

    console.print(f"\n[green]Total processed: {total_ok}[/green]")


# ---------------------------------------------------------------------------
# Step 3+4: Build annotation records
# ---------------------------------------------------------------------------
def _assign_conditions(n_images: int) -> list[str]:
    """
    Assign conditions to n_images images according to CONDITION_DIST.
    Returns a list of condition strings of length n_images.
    """
    conditions: list[str] = []
    for cond, frac in CONDITION_DIST.items():
        count = max(1, round(frac * n_images))
        conditions.extend([cond] * count)

    # trim or pad to exactly n_images
    random.shuffle(conditions)
    if len(conditions) > n_images:
        conditions = conditions[:n_images]
    while len(conditions) < n_images:
        conditions.append(random.choice(list(CONDITION_DIST.keys())))
    random.shuffle(conditions)
    return conditions


def build_sample(
    image_rel: str,
    part_slug: str,
    condition: str,
    sample_id: str,
    confidence: float = 0.95,
) -> dict:
    """
    Build a single InternVL2 conversation sample dict.
    Includes trajectory waypoints for pick-and-place simulation.
    """
    from src.simulation.trajectory import generate_trajectory_for_sample

    part = SLUG_TO_CLASS[part_slug]
    action = apply_sorting_rules(part, condition)

    # Generate pick-and-place trajectory (seeded by sample_id)
    trajectory = generate_trajectory_for_sample(
        part_slug, condition, sample_id=sample_id,
    )

    return {
        "id": sample_id,
        "image": image_rel,
        "part_slug": part_slug,
        "condition": condition,
        "bin": action.bin,
        "priority": action.priority,
        "inspect": action.inspect,
        "trajectory_waypoints": [list(wp) for wp in trajectory.waypoints],
        "pick_point": list(trajectory.pick_point),
        "conversations": [
            {"from": "human", "value": HUMAN_PROMPT},
            {"from": "gpt",   "value": action.to_response_text(confidence)},
        ],
    }


@app.command()
def annotate_auto(
    processed_dir: Path = typer.Option(PROCESSED_DIR, help="Processed images root"),
    out_dir: Path = typer.Option(ANN_DIR, help="Annotations output directory"),
    seed: int = typer.Option(42, help="Random seed for condition assignment"),
):
    """
    Auto-generate action annotations from deterministic sorting rules.

    Assigns conditions based on CONDITION_DIST, then derives bin/priority/inspect
    from the rule engine.  Run 'vla-annotate' afterward for manual condition review.
    """
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    skipped = 0

    for part in track(PART_CLASSES, description="Annotating classes..."):
        class_dir = processed_dir / part.slug
        if not class_dir.exists():
            console.print(f"[yellow]Skipping {part.slug} — no processed images[/yellow]")
            skipped += 1
            continue

        images = sorted(class_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue

        conditions = _assign_conditions(len(images))

        for img_path, condition in zip(images, conditions):
            rel_path = str(img_path.relative_to(processed_dir.parent))
            sample_id = img_path.stem
            sample = build_sample(rel_path, part.slug, condition, sample_id)
            all_samples.append(sample)

    console.print(f"\nBuilt {len(all_samples)} samples ({skipped} classes skipped)")

    # Save full annotation file
    full_path = out_dir / "all_annotations.jsonl"
    with jsonlines.open(full_path, mode="w") as writer:
        writer.write_all(all_samples)
    console.print(f"Saved → {full_path}")

    return all_samples


# ---------------------------------------------------------------------------
# Step 5: Stratified train/val/test split
# ---------------------------------------------------------------------------
@app.command()
def split(
    ann_file: Path = typer.Option(ANN_DIR / "all_annotations.jsonl"),
    out_dir: Path = typer.Option(ANN_DIR),
    train_frac: float = typer.Option(0.70),
    val_frac: float = typer.Option(0.15),
    seed: int = typer.Option(42),
):
    """Split annotations into train/val/test JSONL files (stratified by class)."""
    samples = _load_jsonl(ann_file)
    labels = [s["part_slug"] for s in samples]

    # First split off test set
    train_val, test, labels_tv, _ = train_test_split(
        samples, labels,
        test_size=1.0 - train_frac - val_frac,
        stratify=labels,
        random_state=seed,
    )
    # Split train_val into train and val
    val_ratio_of_tv = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_of_tv,
        stratify=labels_tv,
        random_state=seed,
    )

    splits = {"train": train, "val": val, "test": test}
    for split_name, data in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        _write_jsonl(data, path)
        console.print(f"  {split_name}: {len(data)} samples → {path}")

    _write_meta(splits, out_dir)
    console.print(f"\n[green]Split complete.[/green]")


def _write_meta(splits: dict, out_dir: Path):
    meta = {}
    for split_name, data in splits.items():
        meta[f"vla_autoparts_{split_name}"] = {
            "root": "data/processed/",
            "annotation": f"data/annotations/{split_name}.jsonl",
            "data_augment": split_name == "train",
            "repeat_time": 1,
            "length": len(data),
        }
    path = out_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2))
    console.print(f"  meta.json → {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@app.command()
def validate(
    ann_dir: Path = typer.Option(ANN_DIR),
    processed_dir: Path = typer.Option(PROCESSED_DIR),
):
    """Run QA checks on the dataset."""
    console.print("\n[bold]Running dataset validation...[/bold]")

    errors: list[str] = []
    class_counts: dict[str, int] = defaultdict(int)
    condition_counts: dict[str, int] = defaultdict(int)

    for split_name in ("train", "val", "test"):
        path = ann_dir / f"{split_name}.jsonl"
        if not path.exists():
            console.print(f"[yellow]Split file missing: {path}[/yellow]")
            continue
        samples = _load_jsonl(path)
        for s in samples:
            class_counts[s["part_slug"]] += 1
            condition_counts[s["condition"]] += 1

            # Rule consistency
            rule_errors = validate_annotation(s)
            for e in rule_errors:
                errors.append(f"[{s['id']}] {e}")

            # Image exists
            img_path = processed_dir.parent / s["image"]
            if not img_path.exists():
                errors.append(f"[{s['id']}] Image not found: {s['image']}")

    # Class balance table
    table = Table(title="Class sample counts (train+val+test)")
    table.add_column("Part")
    table.add_column("Count", justify="right")
    table.add_column("Status")
    for part in PART_CLASSES:
        count = class_counts.get(part.slug, 0)
        status = "[green]OK[/green]" if 45 <= count <= 55 else "[red]WARN[/red]"
        table.add_row(part.name, str(count), status)
    console.print(table)

    if errors:
        console.print(f"\n[red]{len(errors)} errors found:[/red]")
        for e in errors[:20]:
            console.print(f"  {e}")
        if len(errors) > 20:
            console.print(f"  ... and {len(errors) - 20} more")
    else:
        console.print("\n[green]All checks passed![/green]")


# ---------------------------------------------------------------------------
# Full pipeline command
# ---------------------------------------------------------------------------
@app.command()
def build(
    rf_key: Optional[str] = typer.Option(None, help="Roboflow API key (skip if images already present)"),
    seed: int = typer.Option(42),
):
    """Run the full pipeline: clean → annotate → split → validate."""
    console.rule("[bold blue]VLA-AutoParts Dataset Pipeline[/bold blue]")

    if rf_key:
        console.rule("Step 1: Download")
        # Download the three main Roboflow datasets
        datasets = [
            ("team-data",            "car-parts-ybiev",                1),
            ("ultralytics",          "carparts-segmentation",          1),
            ("computervision-nyq7i", "car-components-dataset",         1),
        ]
        for ws, proj, ver in datasets:
            try:
                download(rf_key=rf_key, workspace=ws, project=proj, version=ver)
            except Exception as e:
                console.print(f"[yellow]Could not download {proj}: {e}[/yellow]")

    console.rule("Step 2: Clean")
    clean()

    console.rule("Step 3+4: Auto-annotate")
    annotate_auto(seed=seed)

    console.rule("Step 5: Split")
    split(seed=seed)

    console.rule("Step 6: Validate")
    validate()

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print("Next: run [cyan]vla-annotate[/cyan] to manually review/adjust conditions.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader)


def _write_jsonl(data: list[dict], path: Path):
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(data)


def main():
    app()


if __name__ == "__main__":
    main()
