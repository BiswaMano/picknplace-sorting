"""
QLoRA fine-tuning trainer for VLA-AutoParts.

Trains Mini-InternVL-Chat-4B-V1-5 + DisentangledActionHead on the
automotive parts dataset.  Designed to fit within 8GB VRAM (RTX 4060).

Key techniques for 8GB VRAM:
  - 4-bit NF4 quantization (bitsandbytes)
  - QLoRA with r=16
  - Gradient checkpointing
  - Batch size 1 + gradient accumulation (effective batch ~8)
  - bfloat16 compute
  - Frozen vision encoder

Usage:
  python -m src.training.trainer --help
  python -m src.training.trainer train
  python -m src.training.trainer train --epochs 3 --lr 2e-4
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

app = typer.Typer(help="VLA-AutoParts QLoRA trainer")
console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
ANN_DIR     = DATA_DIR / "annotations"
CHECKPOINTS = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

def default_config() -> dict:
    return {
        # Model
        "model_id":           "OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
        "lora_r":             16,
        "lora_alpha":         32,
        "lora_dropout":       0.05,
        # Training
        "epochs":             3,
        "lr":                 2e-4,
        "weight_decay":       0.01,
        "warmup_ratio":       0.05,
        "batch_size":         1,
        "grad_accum_steps":   8,    # effective batch = 8
        "max_grad_norm":      1.0,
        "max_length":         512,
        # Loss
        "language_loss_weight": 1.0,
        "action_loss_weight":   0.5,
        # Hardware
        "fp16":               False,
        "bf16":               True,
        "grad_checkpointing": True,
        # Evaluation
        "eval_steps":         100,
        "save_steps":         200,
        "logging_steps":      10,
    }


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    def __init__(self):
        self.history: list[dict] = []
        self._step_acc: dict = {}

    def update(self, metrics: dict):
        for k, v in metrics.items():
            if k not in self._step_acc:
                self._step_acc[k] = []
            val = v.item() if hasattr(v, "item") else float(v)
            self._step_acc[k].append(val)

    def log_step(self, step: int) -> dict:
        averaged = {k: sum(vs) / len(vs) for k, vs in self._step_acc.items()}
        averaged["step"] = step
        self.history.append(averaged)
        self._step_acc = {}
        return averaged

    def save(self, path: Path):
        path.write_text(json.dumps(self.history, indent=2))


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_action_accuracy(output, action_labels: dict) -> dict:
    pred_class    = output.pred_class
    pred_bin      = output.pred_bin - 1     # 0-indexed for comparison
    pred_priority = output.pred_priority
    pred_inspect  = output.pred_inspect.long()

    acc_class    = (pred_class    == action_labels["labels_class"]).float().mean()
    acc_bin      = (pred_bin      == action_labels["labels_bin"]).float().mean()
    acc_priority = (pred_priority == action_labels["labels_priority"]).float().mean()
    acc_inspect  = (pred_inspect  == action_labels["labels_inspect"].long()).float().mean()

    result = {
        "acc_class":    acc_class,
        "acc_bin":      acc_bin,
        "acc_priority": acc_priority,
        "acc_inspect":  acc_inspect,
    }

    # Trajectory metrics: mean waypoint error (meters)
    if (
        output.pred_trajectory is not None
        and "labels_trajectory" in action_labels
        and "has_trajectory" in action_labels
    ):
        mask = action_labels["has_trajectory"].bool()
        if mask.any():
            pred_wps = output.pred_trajectory[mask].view(-1, 7, 3)
            gt_wps   = action_labels["labels_trajectory"][mask].view(-1, 7, 3)
            # Mean L2 error across all waypoints
            wp_errors = torch.norm(pred_wps - gt_wps, dim=-1)  # (N, 7)
            result["mean_wp_error"]  = wp_errors.mean()
            # Placement error (waypoint index 5 = "place")
            result["place_error"]    = wp_errors[:, 5].mean()

    return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VLATrainer:
    def __init__(self, cfg: dict, out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Device: {self.device}")
        if torch.cuda.is_available():
            console.print(f"GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.tracker = MetricsTracker()

    def setup(self):
        from src.model.vla import load_vla

        console.print("\n[bold]Loading VLA model...[/bold]")
        self.model, self.tokenizer = load_vla(
            model_id=self.cfg["model_id"],
            use_qlora=True,
            lora_r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_alpha"],
            action_loss_weight=self.cfg["action_loss_weight"],
        )

        if self.cfg["grad_checkpointing"]:
            self.model.backbone.gradient_checkpointing_enable()

        # Image processor
        try:
            from transformers import AutoImageProcessor
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.cfg["model_id"], trust_remote_code=True
            )
        except Exception:
            self.image_processor = None
            console.print("[yellow]Image processor not found — vision input disabled[/yellow]")

        self._setup_data()
        self._setup_optimizer()

    def _setup_data(self):
        from src.training.dataset import AutoPartsDataset, collate_fn

        train_ds = AutoPartsDataset(
            jsonl_path=ANN_DIR / "train.jsonl",
            data_root=DATA_DIR,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            max_length=self.cfg["max_length"],
            augment=True,
        )
        val_ds = AutoPartsDataset(
            jsonl_path=ANN_DIR / "val.jsonl",
            data_root=DATA_DIR,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            max_length=self.cfg["max_length"],
            augment=False,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        console.print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    def _setup_optimizer(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        # Only optimize trainable params (LoRA + action head)
        params = [p for p in self.model.parameters() if p.requires_grad]
        console.print(f"Trainable params: {sum(p.numel() for p in params):,}")

        self.optimizer = AdamW(
            params,
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        total_steps = (
            len(self.train_loader) * self.cfg["epochs"]
            // self.cfg["grad_accum_steps"]
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg["lr"],
            total_steps=total_steps,
            pct_start=self.cfg["warmup_ratio"],
            anneal_strategy="cos",
        )

    def train(self):
        self.setup()
        cfg = self.cfg
        global_step = 0
        best_val_loss = float("inf")

        scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"])

        for epoch in range(cfg["epochs"]):
            console.rule(f"[bold]Epoch {epoch + 1}/{cfg['epochs']}[/bold]")
            self.model.train()
            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
                    enabled=cfg["bf16"] or cfg["fp16"],
                ):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch.get("pixel_values"),
                        labels=batch["labels"],
                        action_labels=batch["action_labels"],
                    )

                loss = outputs["loss"] / cfg["grad_accum_steps"]
                scaler.scale(loss).backward()

                # Track metrics
                metrics = {k: v for k, v in outputs.items() if k.startswith("loss")}
                accs = compute_action_accuracy(
                    outputs["action_output"], batch["action_labels"]
                )
                self.tracker.update({**metrics, **accs})

                if (step + 1) % cfg["grad_accum_steps"] == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg["max_grad_norm"]
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg["logging_steps"] == 0:
                        logged = self.tracker.log_step(global_step)
                        self._log(logged, prefix="train")

                    if global_step % cfg["eval_steps"] == 0:
                        val_metrics = self.evaluate()
                        self._log(val_metrics, prefix="val")

                        if val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]
                            self._save_checkpoint("best", global_step)

                    if global_step % cfg["save_steps"] == 0:
                        self._save_checkpoint(f"step_{global_step}", global_step)

            # End-of-epoch save
            self._save_checkpoint(f"epoch_{epoch + 1}", global_step)

        self.tracker.save(self.out_dir / "training_history.json")
        console.print("\n[bold green]Training complete![/bold green]")
        console.print(f"Best val loss: {best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        totals: dict = {}
        n = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
                action_labels=batch["action_labels"],
            )
            accs = compute_action_accuracy(
                outputs["action_output"], batch["action_labels"]
            )
            for k, v in {**outputs, **accs}.items():
                if k in ("action_output", "id"):
                    continue
                val = v.item() if hasattr(v, "item") else float(v)
                totals[k] = totals.get(k, 0.0) + val
            n += 1

        self.model.train()
        return {k: v / n for k, v in totals.items()}

    def _to_device(self, batch: dict) -> dict:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = {sk: sv.to(self.device) for sk, sv in v.items()}
            else:
                result[k] = v
        return result

    def _log(self, metrics: dict, prefix: str = ""):
        parts = []
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
        step = metrics.get("step", "?")
        console.print(f"[{prefix}] step={step} | " + " | ".join(parts))

    def _save_checkpoint(self, tag: str, step: int):
        ckpt_dir = self.out_dir / f"checkpoint_{tag}"
        ckpt_dir.mkdir(exist_ok=True)

        # Save LoRA adapter
        if hasattr(self.model.backbone, "save_pretrained"):
            self.model.backbone.save_pretrained(ckpt_dir / "lora_adapter")

        # Save action head separately (full precision)
        torch.save(
            self.model.action_head.state_dict(),
            ckpt_dir / "action_head.pt",
        )

        # Save config
        (ckpt_dir / "config.json").write_text(json.dumps(self.cfg, indent=2))
        console.print(f"[green]Checkpoint saved → {ckpt_dir}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def train(
    epochs: int   = typer.Option(3),
    lr: float     = typer.Option(2e-4),
    lora_r: int   = typer.Option(16),
    batch_size: int = typer.Option(1),
    grad_accum: int = typer.Option(8),
    out_dir: Path = typer.Option(CHECKPOINTS),
    action_weight: float = typer.Option(0.5, help="Action loss weight"),
):
    """Run QLoRA fine-tuning."""
    cfg = default_config()
    cfg.update({
        "epochs":             epochs,
        "lr":                 lr,
        "lora_r":             lora_r,
        "batch_size":         batch_size,
        "grad_accum_steps":   grad_accum,
        "action_loss_weight": action_weight,
    })

    trainer = VLATrainer(cfg, out_dir)
    trainer.train()


def main():
    app()


if __name__ == "__main__":
    main()
