"""
Gradio-based annotation tool for VLA-AutoParts.

Displays each image and lets the annotator:
  - Select the part condition (dropdown)
  - Review the auto-generated bin/priority/inspect/reason
  - Approve or correct the annotation
  - Navigate forward/back through the image queue

Saves approved annotations to data/annotations/reviewed.jsonl.

Usage:
  python -m src.dataset.annotator
  # or
  vla-annotate
"""

import json
from pathlib import Path

import gradio as gr
import jsonlines

from src.dataset.classes import CONDITIONS, PART_CLASSES, SLUG_TO_CLASS
from src.dataset.sorting_rules import apply_sorting_rules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANN_DIR = DATA_DIR / "annotations"
REVIEWED_PATH = ANN_DIR / "reviewed.jsonl"
ALL_ANN_PATH = ANN_DIR / "all_annotations.jsonl"

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_queue() -> list[dict]:
    """Load the auto-annotated queue, filter out already-reviewed IDs."""
    if not ALL_ANN_PATH.exists():
        return []

    reviewed_ids: set[str] = set()
    if REVIEWED_PATH.exists():
        with jsonlines.open(REVIEWED_PATH) as r:
            for rec in r:
                reviewed_ids.add(rec["id"])

    queue: list[dict] = []
    with jsonlines.open(ALL_ANN_PATH) as r:
        for rec in r:
            if rec["id"] not in reviewed_ids:
                queue.append(rec)
    return queue


def save_reviewed(record: dict):
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(REVIEWED_PATH, mode="a") as w:
        w.write(record)


def recompute_action(part_slug: str, condition: str) -> dict:
    part = SLUG_TO_CLASS[part_slug]
    action = apply_sorting_rules(part, condition)
    return {
        "bin": action.bin,
        "priority": action.priority,
        "inspect": action.inspect,
        "reason": action.reason,
        "response_text": action.to_response_text(),
    }


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    queue: list[dict] = load_queue()
    state = {"idx": 0, "queue": queue, "current": queue[0] if queue else None}

    def get_current():
        idx = state["idx"]
        q = state["queue"]
        if not q or idx >= len(q):
            return None
        return q[idx]

    def render_current(condition_override: str | None = None):
        rec = get_current()
        if rec is None:
            return (
                None,        # image
                "—",         # part name
                "—",         # slug
                CONDITIONS[1],  # condition dropdown
                "—",         # bin
                "—",         # priority
                False,       # inspect
                "Queue is empty or fully reviewed.",  # reason
                "—",         # response preview
                f"0 / {len(state['queue'])}",
            )

        part = SLUG_TO_CLASS.get(rec["part_slug"])
        condition = condition_override or rec["condition"]
        action = recompute_action(rec["part_slug"], condition)

        img_path = PROCESSED_DIR.parent / rec["image"]
        img = str(img_path) if img_path.exists() else None

        progress = f"{state['idx'] + 1} / {len(state['queue'])}"
        return (
            img,
            part.name if part else rec["part_slug"],
            rec["part_slug"],
            condition,
            str(action["bin"]),
            action["priority"],
            action["inspect"],
            action["reason"],
            action["response_text"],
            progress,
        )

    def on_condition_change(condition: str):
        return render_current(condition_override=condition)

    def on_approve(condition: str):
        rec = get_current()
        if rec is None:
            return render_current()
        action = recompute_action(rec["part_slug"], condition)
        approved = {**rec, "condition": condition, **action}
        approved.pop("response_text", None)
        # Update conversations with new condition
        approved["conversations"] = [
            {"from": "human", "value": rec["conversations"][0]["value"]},
            {"from": "gpt",   "value": action["response_text"]},
        ]
        save_reviewed(approved)
        state["idx"] += 1
        return render_current()

    def on_skip():
        state["idx"] += 1
        return render_current()

    def on_prev():
        if state["idx"] > 0:
            state["idx"] -= 1
        return render_current()

    def on_reload():
        state["queue"] = load_queue()
        state["idx"] = 0
        state["current"] = state["queue"][0] if state["queue"] else None
        return render_current()

    # ── Layout ──────────────────────────────────────────────────────────────
    with gr.Blocks(title="VLA-AutoParts Annotation Tool") as demo:
        gr.Markdown("# VLA-AutoParts Annotation Tool")
        gr.Markdown(
            "Review each image, confirm or adjust the condition label, "
            "then click **Approve** to save."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_out = gr.Image(label="Part Image", type="filepath", height=400)
                progress_lbl = gr.Label(label="Progress")

            with gr.Column(scale=1):
                part_name_lbl = gr.Textbox(label="Part Name", interactive=False)
                slug_lbl = gr.Textbox(label="Slug", interactive=False)

                condition_dd = gr.Dropdown(
                    choices=CONDITIONS,
                    label="Condition",
                    value=CONDITIONS[1],
                )

                with gr.Row():
                    bin_lbl      = gr.Textbox(label="Bin",      interactive=False)
                    priority_lbl = gr.Textbox(label="Priority", interactive=False)
                    inspect_cb   = gr.Checkbox(label="Inspect", interactive=False)

                reason_lbl = gr.Textbox(label="Reason", lines=3, interactive=False)
                response_preview = gr.Textbox(
                    label="GPT Response Preview", lines=6, interactive=False
                )

                with gr.Row():
                    prev_btn    = gr.Button("← Prev")
                    skip_btn    = gr.Button("Skip →")
                    approve_btn = gr.Button("✓ Approve", variant="primary")

                reload_btn = gr.Button("Reload Queue")

        outputs = [
            img_out, part_name_lbl, slug_lbl, condition_dd,
            bin_lbl, priority_lbl, inspect_cb, reason_lbl,
            response_preview, progress_lbl,
        ]

        condition_dd.change(on_condition_change, inputs=[condition_dd], outputs=outputs)
        approve_btn.click(on_approve, inputs=[condition_dd], outputs=outputs)
        skip_btn.click(on_skip, outputs=outputs)
        prev_btn.click(on_prev, outputs=outputs)
        reload_btn.click(on_reload, outputs=outputs)

        # Load first record on startup
        demo.load(render_current, outputs=outputs)

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
