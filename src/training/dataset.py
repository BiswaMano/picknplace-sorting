"""
PyTorch Dataset for VLA-AutoParts fine-tuning.

Reads JSONL annotation files and prepares:
  - Tokenized conversation (human turn + GPT response)
  - Pixel values from image (via InternVL2 image processor)
  - Action labels (class, bin, priority, inspect) as tensors

Supports both vision samples (image + text) and Sort Dreaming
text-only samples (image=None).
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import jsonlines

from src.dataset.classes import (
    SLUG_TO_CLASS, PRIORITIES, PRIORITY_IDX, CONDITIONS, CONDITION_IDX, NUM_BINS
)


# ---------------------------------------------------------------------------
# Albumentations augmentation pipeline (training only)
# ---------------------------------------------------------------------------

def build_augmentation():
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import numpy as np

        return A.Compose([
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.HueSaturationValue(p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        ])
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AutoPartsDataset(Dataset):
    """
    Args:
        jsonl_path:       Path to train.jsonl / val.jsonl / test.jsonl
        data_root:        Root directory for resolving image paths
        tokenizer:        HuggingFace tokenizer
        image_processor:  InternVL2 image processor (or None for text-only)
        max_length:       Maximum token length for the conversation
        augment:          Whether to apply image augmentations (training only)
    """

    HUMAN_TEMPLATE = (
        "<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    GPT_TEMPLATE = "{content}<|im_end|>"

    def __init__(
        self,
        jsonl_path: Path,
        data_root: Path,
        tokenizer,
        image_processor=None,
        max_length: int = 512,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.augment = augment
        self.aug_pipeline = build_augmentation() if augment else None

        with jsonlines.open(jsonl_path) as reader:
            self.samples = list(reader)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rec = self.samples[idx]
        return self._process(rec)

    def _process(self, rec: dict) -> dict:
        # ── Action labels ────────────────────────────────────────────────────
        part = SLUG_TO_CLASS.get(rec["part_slug"])
        class_label    = part.idx if part else 0
        bin_label      = max(0, rec["bin"] - 1)          # 0-indexed
        priority_label = PRIORITY_IDX.get(rec["priority"], 1)
        inspect_label  = float(rec.get("inspect", False))

        # Trajectory labels (7 waypoints x 3 coords = 21 floats)
        traj_wps = rec.get("trajectory_waypoints")
        if traj_wps is not None:
            flat_traj = [c for wp in traj_wps for c in wp]
            labels_trajectory = torch.tensor(flat_traj, dtype=torch.float)
            has_trajectory = torch.tensor(1.0)
        else:
            labels_trajectory = torch.zeros(21, dtype=torch.float)
            has_trajectory = torch.tensor(0.0)

        action_labels = {
            "labels_class":      torch.tensor(class_label,    dtype=torch.long),
            "labels_bin":        torch.tensor(bin_label,      dtype=torch.long),
            "labels_priority":   torch.tensor(priority_label, dtype=torch.long),
            "labels_inspect":    torch.tensor(inspect_label,  dtype=torch.float),
            "labels_trajectory": labels_trajectory,
            "has_trajectory":    has_trajectory,
        }

        # ── Text tokenization ────────────────────────────────────────────────
        human_val = rec["conversations"][0]["value"]
        gpt_val   = rec["conversations"][1]["value"]

        # Remove <image> token placeholder if no image
        has_image = rec.get("image") is not None
        if not has_image:
            human_val = human_val.replace("<image>\n", "").replace("<image>", "")

        human_text = self.HUMAN_TEMPLATE.format(content=human_val)
        gpt_text   = self.GPT_TEMPLATE.format(content=gpt_val)
        full_text  = human_text + gpt_text

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Create labels: mask the human turn with -100 (don't compute loss on it)
        human_len = len(self.tokenizer(human_text, add_special_tokens=False)["input_ids"])
        labels = input_ids.clone()
        labels[:human_len] = -100

        # ── Image processing ─────────────────────────────────────────────────
        pixel_values = None
        if has_image and self.image_processor is not None:
            img_path = self.data_root / rec["image"]
            try:
                img = Image.open(img_path).convert("RGB")
                if self.aug_pipeline is not None:
                    import numpy as np
                    img_np = np.array(img)
                    img_np = self.aug_pipeline(image=img_np)["image"]
                    img = Image.fromarray(img_np)
                pv = self.image_processor(images=img, return_tensors="pt")
                pixel_values = pv["pixel_values"].squeeze(0)
            except Exception:
                pixel_values = None

        result = {
            "id":            rec["id"],
            "input_ids":     input_ids,
            "attention_mask": attention_mask,
            "labels":        labels,
            "action_labels": action_labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles optional pixel_values."""
    keys = batch[0].keys()
    collated: dict = {}

    for key in keys:
        if key == "action_labels":
            collated["action_labels"] = {
                k: torch.stack([b["action_labels"][k] for b in batch])
                for k in batch[0]["action_labels"]
            }
        elif key == "id":
            collated["id"] = [b["id"] for b in batch]
        elif key == "pixel_values":
            # Only include if all samples have pixel_values
            pvs = [b.get("pixel_values") for b in batch]
            if all(pv is not None for pv in pvs):
                collated["pixel_values"] = torch.stack(pvs)
        else:
            collated[key] = torch.stack([b[key] for b in batch])

    return collated
