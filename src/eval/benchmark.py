"""
Evaluation suite for VLA-AutoParts.

1. Standard benchmark: classification accuracy, bin accuracy, priority accuracy,
   inspect F1 — on the held-out test set.

2. Sort Dreaming evaluation: text-only accuracy on the 500-sample
   Sort Dreaming set — measures language-action alignment.

3. Baseline comparison: runs zero-shot InternVL2 (no fine-tuning) on the
   same test set via text parsing of generated responses.

Usage:
  python -m src.eval.benchmark --checkpoint checkpoints/checkpoint_best
  python -m src.eval.benchmark --sort-dreaming-only
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
import jsonlines

from src.dataset.classes import (
    SLUG_TO_CLASS, IDX_TO_CLASS, PRIORITIES, PRIORITY_IDX,
    CONDITIONS, CONDITION_IDX, CLASS_SLUGS,
)
from src.dataset.sorting_rules import apply_sorting_rules

app = typer.Typer()
console = Console()

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
ANN_DIR     = DATA_DIR / "annotations"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Response parser — extracts structured fields from generated text
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict:
    """
    Parse a VLA model response string into structured fields.

    Expected format:
      Part: <slug>
      Condition: <condition>
      Confidence: <float>
      Action: bin=<int>, priority=<str>, inspect=<bool>
      Reason: <text>
    """
    result = {}

    m = re.search(r"Part:\s*(\S+)", text)
    if m:
        result["part_slug"] = m.group(1).strip()

    m = re.search(r"Condition:\s*(\S+)", text)
    if m:
        result["condition"] = m.group(1).strip()

    m = re.search(r"Confidence:\s*([\d.]+)", text)
    if m:
        result["confidence"] = float(m.group(1))

    m = re.search(r"bin=(\d+)", text)
    if m:
        result["bin"] = int(m.group(1))

    m = re.search(r"priority=(\w+)", text)
    if m:
        result["priority"] = m.group(1).strip()

    m = re.search(r"inspect=(true|false)", text, re.IGNORECASE)
    if m:
        result["inspect"] = m.group(1).lower() == "true"

    m = re.search(r"Reason:\s*(.+)", text, re.DOTALL)
    if m:
        result["reason"] = m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class EvalMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.correct_class    = 0
        self.correct_bin      = 0
        self.correct_priority = 0
        self.tp_inspect = self.fp_inspect = self.fn_inspect = self.tn_inspect = 0

    def update_from_action_head(self, output, action_labels: dict):
        B = action_labels["labels_class"].size(0)
        self.n += B
        self.correct_class    += (output.pred_class    == action_labels["labels_class"]).sum().item()
        self.correct_bin      += ((output.pred_bin - 1) == action_labels["labels_bin"]).sum().item()
        self.correct_priority += (output.pred_priority == action_labels["labels_priority"]).sum().item()

        pred_i  = output.pred_inspect.long()
        true_i  = action_labels["labels_inspect"].long()
        self.tp_inspect += (pred_i & true_i).sum().item()
        self.fp_inspect += (pred_i & ~true_i.bool()).sum().item()
        self.fn_inspect += (~pred_i.bool() & true_i.bool()).sum().item()
        self.tn_inspect += (~pred_i.bool() & ~true_i.bool()).sum().item()

    def update_from_parsed(self, parsed: dict, ground_truth: dict):
        self.n += 1
        if parsed.get("part_slug") == ground_truth.get("part_slug"):
            self.correct_class += 1
        if parsed.get("bin") == ground_truth.get("bin"):
            self.correct_bin += 1
        if parsed.get("priority") == ground_truth.get("priority"):
            self.correct_priority += 1

        pred_i = bool(parsed.get("inspect", False))
        true_i = bool(ground_truth.get("inspect", False))
        if pred_i and true_i:      self.tp_inspect += 1
        elif pred_i and not true_i: self.fp_inspect += 1
        elif not pred_i and true_i: self.fn_inspect += 1
        else:                       self.tn_inspect += 1

    def summary(self) -> dict:
        if self.n == 0:
            return {}
        prec = self.tp_inspect / (self.tp_inspect + self.fp_inspect + 1e-9)
        rec  = self.tp_inspect / (self.tp_inspect + self.fn_inspect + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        return {
            "n":                self.n,
            "acc_class":        self.correct_class    / self.n,
            "acc_bin":          self.correct_bin      / self.n,
            "acc_priority":     self.correct_priority / self.n,
            "inspect_precision": prec,
            "inspect_recall":   rec,
            "inspect_f1":       f1,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class VLAEvaluator:
    def __init__(self, checkpoint_path: Optional[Path], device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        from src.model.vla import load_vla
        ckpt = str(self.checkpoint_path) if self.checkpoint_path else None
        self.model, self.tokenizer = load_vla(checkpoint_path=ckpt)
        self.model.eval()
        console.print(f"[green]Model loaded on {self.device}[/green]")

    # ── Action-head evaluation ─────────────────────────────────────────────

    def eval_test_set(self) -> dict:
        from src.training.dataset import AutoPartsDataset, collate_fn
        from transformers import AutoImageProcessor

        image_processor = None
        try:
            from src.model.vla import INTERNVL_MODEL_ID
            image_processor = AutoImageProcessor.from_pretrained(
                INTERNVL_MODEL_ID, trust_remote_code=True
            )
        except Exception:
            pass

        ds = AutoPartsDataset(
            jsonl_path=ANN_DIR / "test.jsonl",
            data_root=DATA_DIR,
            tokenizer=self.tokenizer,
            image_processor=image_processor,
            max_length=512,
            augment=False,
        )
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

        metrics = EvalMetrics()
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor)
                        else ({sk: sv.to(self.device) for sk, sv in v.items()} if isinstance(v, dict) else v))
                    for k, v in batch.items()
                }
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    action_labels=batch["action_labels"],
                )
                metrics.update_from_action_head(
                    outputs["action_output"], batch["action_labels"]
                )

        return metrics.summary()

    # ── Sort Dreaming evaluation ───────────────────────────────────────────

    def eval_sort_dreaming(self) -> dict:
        sd_path = ANN_DIR / "sort_dreaming.jsonl"
        if not sd_path.exists():
            console.print("[yellow]Sort Dreaming file not found. Run sort_dreaming.py first.[/yellow]")
            return {}

        with jsonlines.open(sd_path) as reader:
            samples = list(reader)

        metrics = EvalMetrics()

        for rec in samples:
            human_text = rec["conversations"][0]["value"]
            gt = {
                "part_slug": rec["part_slug"],
                "bin":       rec["bin"],
                "priority":  rec["priority"],
                "inspect":   rec["inspect"],
            }

            # Tokenize and generate
            inputs = self.tokenizer(
                human_text, return_tensors="pt", max_length=256,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.backbone.generate(
                    **inputs, max_new_tokens=100, do_sample=False
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            parsed = parse_response(text)
            metrics.update_from_parsed(parsed, gt)

        return metrics.summary()

    # ── Baseline: zero-shot InternVL2 (no fine-tuning) ────────────────────

    def eval_baseline_zero_shot(self) -> dict:
        """
        Run zero-shot InternVL2 (stock weights, no LoRA) on a 100-sample
        subset of the test set for baseline comparison.
        """
        from transformers import AutoTokenizer, AutoModel

        console.print("[cyan]Loading baseline (zero-shot) model...[/cyan]")
        from src.model.vla import INTERNVL_MODEL_ID, build_qlora_config
        import torch

        baseline_tokenizer = AutoTokenizer.from_pretrained(
            INTERNVL_MODEL_ID, trust_remote_code=True
        )
        baseline_model = AutoModel.from_pretrained(
            INTERNVL_MODEL_ID,
            trust_remote_code=True,
            quantization_config=build_qlora_config(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        baseline_model.eval()

        sd_path = ANN_DIR / "sort_dreaming.jsonl"
        with jsonlines.open(sd_path) as reader:
            samples = list(reader)[:100]   # 100-sample subset for speed

        metrics = EvalMetrics()
        for rec in samples:
            human_text = rec["conversations"][0]["value"]
            gt = {
                "part_slug": rec["part_slug"],
                "bin":       rec["bin"],
                "priority":  rec["priority"],
                "inspect":   rec["inspect"],
            }
            inputs = baseline_tokenizer(
                human_text, return_tensors="pt", max_length=256, truncation=True
            ).to(self.device)

            with torch.no_grad():
                generated = baseline_model.generate(
                    **inputs, max_new_tokens=100, do_sample=False
                )
            text = baseline_tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            parsed = parse_response(text)
            metrics.update_from_parsed(parsed, gt)

        return metrics.summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(title: str, results: dict):
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for k, v in results.items():
        if k == "n":
            table.add_row("samples", str(v))
        else:
            color = "green" if isinstance(v, float) and v >= 0.8 else "yellow"
            table.add_row(k, f"[{color}]{v:.4f}[/{color}]")

    console.print(table)


@app.command()
def evaluate(
    checkpoint: Optional[Path] = typer.Option(None, help="Checkpoint dir (None = current weights)"),
    test_set: bool = typer.Option(True,  help="Evaluate on test.jsonl"),
    sort_dreaming: bool = typer.Option(True,  help="Evaluate Sort Dreaming set"),
    baseline: bool = typer.Option(False, help="Also run zero-shot baseline"),
    out_dir: Path = typer.Option(RESULTS_DIR),
):
    """Run the full evaluation suite."""
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = VLAEvaluator(checkpoint_path=checkpoint)
    evaluator.load_model()
    all_results: dict = {}

    if test_set:
        console.rule("[bold]Test Set Evaluation[/bold]")
        res = evaluator.eval_test_set()
        _print_results("Test Set — Action Head", res)
        all_results["test_set"] = res

    if sort_dreaming:
        console.rule("[bold]Sort Dreaming Evaluation[/bold]")
        res = evaluator.eval_sort_dreaming()
        _print_results("Sort Dreaming — Language-Action Alignment", res)
        all_results["sort_dreaming"] = res

    if baseline:
        console.rule("[bold]Baseline (Zero-Shot)[/bold]")
        res = evaluator.eval_baseline_zero_shot()
        _print_results("Zero-Shot InternVL2 (Baseline)", res)
        all_results["baseline_zero_shot"] = res

    # Save
    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    console.print(f"\n[green]Results saved → {out_path}[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
