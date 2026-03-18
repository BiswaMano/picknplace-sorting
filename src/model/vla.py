"""
VLA-AutoParts: Full Vision-Language-Action model.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Mini-InternVL-Chat-4B-V1-5                 │
  │  ┌──────────────┐   ┌──────────────────┐    │
  │  │ InternViT    │   │ InternLM-2 (LLM) │    │
  │  │ 300M         │──▶│ 4B (QLoRA)       │    │
  │  │ (frozen)     │   │                  │    │
  │  └──────────────┘   └────────┬─────────┘    │
  │                              │ hidden states │
  └──────────────────────────────┼──────────────┘
                                 ▼
                     DisentangledActionHead
                    (classification + sorting)

The vision encoder (InternViT-300M) is kept frozen.
The LLM (InternLM-2) is adapted with QLoRA (4-bit NF4).
The action head is trained from scratch (full precision).

This file also provides:
  - VLAAutoPartsModel: the full model class
  - load_vla(): convenience loader
  - QLoRA config builder for 8GB VRAM
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from src.model.action_head import DisentangledActionHead, ActionHeadLoss, ActionHeadOutput
from src.dataset.classes import (
    PART_CLASSES, IDX_TO_CLASS, PRIORITIES, CONDITIONS,
    PRIORITY_IDX, CONDITION_IDX,
)
from src.dataset.sorting_rules import apply_sorting_rules


# ---------------------------------------------------------------------------
# Model ID
# ---------------------------------------------------------------------------
INTERNVL_MODEL_ID = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"

# Hidden dimension of InternLM-2 4B
INTERNVL_D_MODEL = 4096

# Action head hidden dim — tuned for 8GB VRAM
ACTION_HEAD_D_HIDDEN = 512


# ---------------------------------------------------------------------------
# QLoRA config
# ---------------------------------------------------------------------------

def build_qlora_config():
    """Build BitsAndBytesConfig for 4-bit NF4 quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("transformers>=4.40.0 required")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4GB
    )


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Build LoRA config targeting LLM attention and MLP layers."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError("peft>=0.10.0 required")

    # Target modules for InternLM-2 attention + MLP
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )


# ---------------------------------------------------------------------------
# VLA model
# ---------------------------------------------------------------------------

class VLAAutoPartsModel(nn.Module):
    """
    Mini-InternVL-Chat-4B-V1-5 + QLoRA + DisentangledActionHead.

    The model operates in two modes:
      1. Language mode (training/generation): standard causal LM loss on
         the conversation response token sequence.
      2. Action mode: the action head takes the final hidden state and
         produces structured action predictions.

    During training both losses are combined:
      L_total = L_language + alpha * L_action
    """

    def __init__(
        self,
        backbone,           # loaded InternVLChatModel
        action_head: DisentangledActionHead,
        action_loss_fn: ActionHeadLoss,
        language_loss_weight: float = 1.0,
        action_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.action_loss_fn = action_loss_fn
        self.language_loss_weight = language_loss_weight
        self.action_loss_weight = action_loss_weight

    def forward(
        self,
        # Standard InternVL2 inputs
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        pixel_values:    Optional[torch.Tensor] = None,
        labels:          Optional[torch.Tensor] = None,
        # Action head supervision
        action_labels:   Optional[dict] = None,
        output_hidden_states: bool = True,
    ) -> dict:
        """
        Args:
            input_ids, attention_mask, pixel_values, labels:
                Standard InternVL2 inputs.
            action_labels: dict with keys:
                labels_class    (B,) long
                labels_bin      (B,) long  (0-indexed)
                labels_priority (B,) long
                labels_inspect  (B,) float
            output_hidden_states: always True during training.

        Returns:
            dict with keys: loss, loss_language, loss_action, action_output
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        result = {}

        # Language loss
        loss_language = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
        result["loss_language"] = loss_language

        # Action head
        hidden_states = outputs.hidden_states[-1]  # last layer: (B, S, D)
        action_output: ActionHeadOutput = self.action_head(hidden_states, attention_mask)
        result["action_output"] = action_output

        if action_labels is not None:
            action_losses = self.action_loss_fn(action_output, **action_labels)
            result["loss_action"] = action_losses["loss"]
            result.update({f"action_{k}": v for k, v in action_losses.items()})
        else:
            result["loss_action"] = torch.tensor(0.0, device=loss_language.device)

        total = (
            self.language_loss_weight * loss_language +
            self.action_loss_weight   * result["loss_action"]
        )
        result["loss"] = total

        return result

    @torch.inference_mode()
    def predict(
        self,
        pixel_values: Optional[torch.Tensor],
        input_ids:    torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config=None,
        tokenizer=None,
    ) -> dict:
        """
        Run inference: generate language response + action head predictions.

        Returns:
            {
                "text":          generated text string,
                "pred_class":    int (0-49),
                "pred_class_name": str,
                "pred_bin":      int (1-10),
                "pred_priority": str,
                "pred_inspect":  bool,
            }
        """
        from transformers import GenerationConfig

        gen_config = generation_config or GenerationConfig(
            max_new_tokens=150,
            do_sample=False,
            temperature=1.0,
        )

        # Generate text
        generated_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            generation_config=gen_config,
        )

        text = ""
        if tokenizer is not None:
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Action head on the input context (not generated tokens)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        action_out = self.action_head(hidden, attention_mask)

        pred_class_idx = action_out.pred_class[0].item()
        pred_bin       = action_out.pred_bin[0].item()
        pred_priority_idx = action_out.pred_priority[0].item()
        pred_inspect   = action_out.pred_inspect[0].item()

        result = {
            "text":             text,
            "pred_class":       pred_class_idx,
            "pred_class_name":  IDX_TO_CLASS[pred_class_idx].name,
            "pred_bin":         pred_bin,
            "pred_priority":    PRIORITIES[pred_priority_idx],
            "pred_inspect":     bool(pred_inspect),
        }

        # Include trajectory if the action head predicts it
        if action_out.pred_waypoints is not None:
            result["pred_trajectory"] = action_out.pred_waypoints[0].tolist()

        return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_vla(
    model_id: str = INTERNVL_MODEL_ID,
    checkpoint_path: Optional[str] = None,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    action_loss_weight: float = 0.5,
    device_map: str = "auto",
) -> tuple[VLAAutoPartsModel, object]:
    """
    Load the backbone, apply QLoRA, attach the action head.

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    console_print = lambda s: print(s)

    console_print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )

    console_print(f"Loading backbone {'with QLoRA' if use_qlora else 'in full precision'}...")
    load_kwargs = dict(
        trust_remote_code=True,
        device_map=device_map,
        output_hidden_states=True,
    )
    if use_qlora:
        load_kwargs["quantization_config"] = build_qlora_config()
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float16

    backbone = AutoModel.from_pretrained(model_id, **load_kwargs)

    if use_qlora:
        backbone = prepare_model_for_kbit_training(backbone)
        lora_cfg = build_lora_config(r=lora_r, lora_alpha=lora_alpha)
        backbone = get_peft_model(backbone, lora_cfg)
        backbone.print_trainable_parameters()

    # Freeze vision encoder — only fine-tune LLM
    for name, param in backbone.named_parameters():
        if "vision_model" in name or "vit" in name.lower():
            param.requires_grad_(False)

    # Build and attach action head
    action_head = DisentangledActionHead(
        d_model=INTERNVL_D_MODEL,
        d_hidden=ACTION_HEAD_D_HIDDEN,
    )
    action_loss_fn = ActionHeadLoss(lambda_inspect=2.0)

    model = VLAAutoPartsModel(
        backbone=backbone,
        action_head=action_head,
        action_loss_fn=action_loss_fn,
        action_loss_weight=action_loss_weight,
    )

    if checkpoint_path:
        console_print(f"Loading checkpoint from {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model, tokenizer
