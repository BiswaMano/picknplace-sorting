"""
VLA-AutoParts Demo — Gradio web interface.

Upload an image of an automotive part → the VLA model identifies it,
assesses its condition, and outputs the sorting action (bin, priority, inspect).

Usage:
  python -m demo.app
  python -m demo.app --checkpoint checkpoints/checkpoint_best
  vla-demo
"""

import json
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import typer

app_cli = typer.Typer()

ROOT = Path(__file__).resolve().parents[1]

# Bin descriptions for display
BIN_DESCRIPTIONS = {
    1:  "Bin 1 — Braking (Premium)",
    2:  "Bin 2 — Braking (Standard)",
    3:  "Bin 3 — Braking (Line Components)",
    4:  "Bin 4 — Suspension (Safety-Critical)",
    5:  "Bin 5 — Suspension (Ride Components)",
    6:  "Bin 6 — Engine (Sensors / Ignition)",
    7:  "Bin 7 — Engine (Mechanical)",
    8:  "Bin 8 — Electrical / Climate",
    9:  "Bin 9 — Accessories / Motors",
    10: "Bin 10 — REJECT / Inspection Hold",
}

PRIORITY_COLORS = {
    "low":    "🟢 Low",
    "normal": "🟡 Normal",
    "high":   "🟠 High",
    "urgent": "🔴 URGENT",
}


# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def get_model(checkpoint_path: Optional[str] = None):
    key = checkpoint_path or "default"
    if key not in _model_cache:
        from src.model.vla import load_vla
        model, tokenizer = load_vla(
            checkpoint_path=checkpoint_path,
            use_qlora=True,
        )
        model.eval()
        _model_cache[key] = (model, tokenizer)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def run_inference(image, checkpoint_path: Optional[str] = None) -> tuple:
    """
    Args:
        image: PIL Image from Gradio
        checkpoint_path: optional path to fine-tuned checkpoint

    Returns:
        Tuple of outputs for Gradio components.
    """
    if image is None:
        return "—", "—", "—", "—", "—", "No image provided."

    try:
        model, tokenizer = get_model(checkpoint_path)
        device = next(model.parameters()).device

        # Build prompt
        prompt = (
            "<|im_start|>user\n"
            "<image>\nIdentify this automotive part, assess its condition, "
            "and provide the sorting action.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Tokenize text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Process image
        try:
            from transformers import AutoImageProcessor
            from src.model.vla import INTERNVL_MODEL_ID
            proc = AutoImageProcessor.from_pretrained(
                INTERNVL_MODEL_ID, trust_remote_code=True
            )
            pv = proc(images=image, return_tensors="pt")["pixel_values"].to(device)
        except Exception:
            pv = None

        result = model.predict(
            pixel_values=pv,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
        )

        # Format outputs
        part_name    = result["pred_class_name"]
        bin_str      = BIN_DESCRIPTIONS.get(result["pred_bin"], f"Bin {result['pred_bin']}")
        priority_str = PRIORITY_COLORS.get(result["pred_priority"], result["pred_priority"])
        inspect_str  = "⚠️ YES — Human inspection required" if result["pred_inspect"] else "✅ No inspection needed"
        response_txt = result.get("text", "")

        # Color-code the bin for severely damaged (bin 10)
        if result["pred_bin"] == 10:
            bin_str = f"🚫 {bin_str}"

        return part_name, bin_str, priority_str, inspect_str, response_txt, "OK"

    except Exception as e:
        return "Error", "—", "—", "—", "—", f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(checkpoint_path: Optional[str] = None) -> gr.Blocks:

    def predict_fn(image):
        return run_inference(image, checkpoint_path)

    with gr.Blocks(
        title="VLA-AutoParts — Automotive Parts Sorting",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
# VLA-AutoParts: Vision-Language-Action Model
### Automotive Parts Recognition & Sorting
*Mini-InternVL-Chat-4B-V1-5 + QLoRA + Disentangled Action Head*
---
Upload an image of an automotive part. The model will:
1. **Identify** the part (50-class recognition)
2. **Assess** its condition (new → severely damaged)
3. **Output** the sorting action (bin, priority, inspection flag)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Part Image",
                    height=400,
                )
                predict_btn = gr.Button("Analyse Part", variant="primary", size="lg")

                gr.Examples(
                    examples=[
                        [str(ROOT / "data" / "processed" / "brake_caliper" / "brake_caliper_good_0000.jpg")],
                    ],
                    inputs=[image_input],
                    label="Example Images (requires dataset to be built first)",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Identification")
                part_name_out = gr.Textbox(label="Part Identified", interactive=False)

                gr.Markdown("### Sorting Action")
                bin_out      = gr.Textbox(label="Bin Assignment", interactive=False)
                priority_out = gr.Textbox(label="Priority",       interactive=False)
                inspect_out  = gr.Textbox(label="Inspection Flag", interactive=False)

                gr.Markdown("### Model Response")
                response_out = gr.Textbox(
                    label="Full VLA Response", lines=8, interactive=False
                )
                status_out = gr.Textbox(label="Status", interactive=False, visible=False)

        predict_btn.click(
            predict_fn,
            inputs=[image_input],
            outputs=[part_name_out, bin_out, priority_out, inspect_out, response_out, status_out],
        )
        image_input.change(
            predict_fn,
            inputs=[image_input],
            outputs=[part_name_out, bin_out, priority_out, inspect_out, response_out, status_out],
        )

        gr.Markdown(
            """
---
### Bin Layout
| Bins | Category |
|------|----------|
| 1–3  | Braking components |
| 4–5  | Suspension |
| 6–7  | Engine / Drivetrain |
| 8–9  | Electrical / Accessories |
| 10   | Reject / Inspection Hold |

*Inspired by SimLingo (Renz et al., CVPR 2025)*
            """
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app_cli.command()
def launch(
    checkpoint: Optional[Path] = typer.Option(None, help="Path to checkpoint directory"),
    port: int = typer.Option(7861),
    share: bool = typer.Option(False, help="Create public Gradio link"),
):
    """Launch the VLA-AutoParts demo."""
    ckpt = str(checkpoint) if checkpoint else None
    demo = build_demo(checkpoint_path=ckpt)
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


def main():
    app_cli()


if __name__ == "__main__":
    main()
