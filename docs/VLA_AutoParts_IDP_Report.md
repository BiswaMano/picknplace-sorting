# VLA-AutoParts: Vision-Language-Action Model for Automotive Parts Recognition and Sorting

## Industry Development Program — Technical Report

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Background and Related Work](#3-background-and-related-work)
   - 3.1 Vision-Language Models (VLMs)
   - 3.2 From VLMs to VLAs — The Missing Action Link
   - 3.3 SimLingo and the Disentangled Action Head
   - 3.4 Positioning of This Work
4. [System Architecture](#4-system-architecture)
   - 4.1 Architecture Overview
   - 4.2 Vision Encoder — InternViT-300M
   - 4.3 Language Model — InternLM-2 (4B) with QLoRA
   - 4.4 Disentangled Action Head
   - 4.5 Trajectory Prediction Branch
   - 4.6 Dual-Loss Training Objective
5. [Dataset Construction](#5-dataset-construction)
   - 5.1 Part Taxonomy — 50 Classes
   - 5.2 Condition Assessment Framework
   - 5.3 Deterministic Sorting Rule Engine
   - 5.4 Image Acquisition Pipeline
   - 5.5 Annotation and Conversation Format
   - 5.6 Sort Dreaming — Language-Action Alignment
   - 5.7 Trajectory Ground Truth Generation
   - 5.8 Data Splits and Validation
6. [Training Methodology](#6-training-methodology)
   - 6.1 QLoRA Fine-Tuning Strategy
   - 6.2 Memory-Efficient Training for 8GB VRAM
   - 6.3 Multi-Task Loss Function
   - 6.4 Training Configuration
7. [Evaluation Framework](#7-evaluation-framework)
   - 7.1 Action Head Metrics
   - 7.2 Sort Dreaming Evaluation
   - 7.3 Zero-Shot Baseline Comparison
   - 7.4 Trajectory Evaluation Metrics
8. [Simulated Pick-and-Place Environment](#8-simulated-pick-and-place-environment)
   - 8.1 Workspace Layout
   - 8.2 Trajectory Generation
   - 8.3 Visualization
9. [Future Enhancements](#9-future-enhancements)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Abstract

This report presents **VLA-AutoParts**, a Vision-Language-Action (VLA) model for automated recognition and sorting of automotive parts. The system extends the Vision-Language Model (VLM) paradigm — widely adopted for image understanding and visual question answering — into the action domain, enabling the model to not only *perceive* and *describe* an automotive part, but also to *decide* and *act* upon it by predicting structured sorting actions and robotic pick-and-place trajectories.

The architecture is built on **Mini-InternVL-Chat-4B-V1-5**, a 4-billion parameter multimodal model, adapted via **QLoRA** (Quantized Low-Rank Adaptation) for parameter-efficient fine-tuning within an 8GB VRAM budget. A custom **Disentangled Action Head**, inspired by the SimLingo framework (Renz et al., CVPR 2025), is appended to the language model backbone, producing structured outputs: 50-class part identification, 10-bin sorting assignment, 4-level priority classification, binary inspection flag, and 7-waypoint pick-and-place trajectory prediction.

The project introduces a deterministic sorting rule engine, a comprehensive 50-class automotive parts taxonomy with safety-critical annotations, and a Sort Dreaming evaluation protocol adapted from SimLingo's Action Dreaming, which tests language-to-action alignment without visual input.

---

## 2. Introduction and Motivation

### 2.1 The Problem

Automotive parts recycling and remanufacturing facilities process thousands of components daily. Each part must be:

1. **Identified** — What is this part? (brake caliper, oil filter, spark plug, ...)
2. **Assessed** — What condition is it in? (new, worn, corroded, damaged, ...)
3. **Sorted** — Which bin does it go to, at what priority, and does a human need to inspect it?

Today, this process relies on trained human operators who visually inspect each part and make sorting decisions. This is slow, inconsistent, and does not scale. Parts with safety-critical implications (brake components, steering linkages, timing belts) require special handling that is easy to miss under production pressure.

### 2.2 The Vision

VLA-AutoParts proposes an AI system that automates this entire pipeline. A camera observes each part as it arrives on a conveyor belt. The VLA model:

- **Sees** the part (vision encoder processes the image)
- **Understands** it (language model identifies the part and reasons about its condition)
- **Decides** what to do (action head predicts bin, priority, and inspection requirements)
- **Acts** on it (trajectory branch predicts the pick-and-place waypoints for a robotic arm)

This transforms a traditional classification problem into a full perception-reasoning-action loop — the defining characteristic of a Vision-Language-Action model.

### 2.3 Why VLA Over Traditional Approaches

| Approach | Identifies Part | Assesses Condition | Decides Action | Explains Decision | Predicts Trajectory |
|----------|:-:|:-:|:-:|:-:|:-:|
| CNN Classifier | Yes | No | No | No | No |
| Object Detection (YOLO) | Yes | No | No | No | No |
| VLM (SigLIP, PaLiGemma) | Yes | Yes (text) | No | Yes | No |
| **VLA (This Work)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

A CNN can tell you "this is a brake caliper." A VLM can tell you "this is a corroded brake caliper." But only a VLA can tell you "this is a corroded brake caliper — route it to Bin 1, flag it as high priority, require human inspection, and here is the trajectory to move it there" — all in a single forward pass.

---

## 3. Background and Related Work

### 3.1 Vision-Language Models (VLMs)

Vision-Language Models combine a vision encoder with a large language model to enable multimodal understanding. The evolution:

- **CLIP** (Radford et al., 2021) — Contrastive learning between image and text embeddings. Demonstrated that vision and language can share a representation space.
- **SigLIP** (Zhai et al., 2023) — Replaced CLIP's softmax loss with a sigmoid loss, improving efficiency and scalability. Used as the vision encoder in many subsequent models.
- **PaLiGemma** (Google, 2024) — Combined SigLIP vision encoder with Gemma LLM for visual question answering and image captioning.
- **InternVL2** (Chen et al., 2024) — Combined InternViT vision encoder with InternLM-2 language model. Achieved state-of-the-art performance on multimodal benchmarks. Available in sizes from 1B to 108B parameters.

These models excel at *perception* (seeing) and *language* (describing), but they lack an *action* output. When asked "what should I do with this part?", they can generate a text answer, but the answer is unstructured — it must be parsed, may contain hallucinations, and cannot directly drive a robotic system.

### 3.2 From VLMs to VLAs — The Missing Action Link

Vision-Language-Action models extend VLMs by adding a structured action output. The key papers:

- **RT-2** (Brohan et al., 2023) — Google DeepMind's seminal work. Showed that a VLM (PaLI-X) can be fine-tuned to output robotic actions by tokenizing actions as text. The model generates action tokens like `[x=0.3, y=0.1, z=0.5, grip=close]` as part of its text output.

- **OpenVLA** (Kim et al., 2024) — Open-source VLA built on Prismatic VLM + Llama 2 backbone. Made VLA research accessible outside of large corporate labs. Demonstrated that VLAs can be fine-tuned for specific robotic tasks with modest compute.

- **SimLingo** (Renz et al., CVPR 2025) — The primary inspiration for this project. SimLingo introduced a **disentangled action head** for autonomous driving, separating perception from action prediction. Key insight: keeping the classification and action branches separate prevents gradient interference and improves action accuracy.

### 3.3 SimLingo and the Disentangled Action Head

SimLingo's architecture is the direct inspiration for VLA-AutoParts. The parallels:

| SimLingo (Driving) | VLA-AutoParts (Sorting) |
|---------------------|--------------------------|
| Dashcam video input | Part image input |
| Scene narration (language) | Part description and sorting rationale (language) |
| **Disentangled Action Head:** | **Disentangled Action Head:** |
| — Scene classification branch | — 50-class part classification branch |
| — Driving action branch (steering, throttle, brake) | — Sorting action branch (bin, priority, inspect) |
| — Trajectory branch (future waypoints) | — Trajectory branch (pick-and-place waypoints) |
| Action Dreaming evaluation | Sort Dreaming evaluation |

SimLingo demonstrated two critical findings that this project adopts:

1. **Disentanglement improves accuracy**: Keeping classification and action branches separate (they share a projection layer but have independent downstream networks) prevents the classification gradient from corrupting the action predictions and vice versa.

2. **Action Dreaming enables language-action evaluation**: By generating text-only hypothetical scenarios (no image), the model's ability to map language descriptions to correct actions can be tested independently of visual perception. This project adapts this as **Sort Dreaming**.

### 3.4 Positioning of This Work

VLA-AutoParts occupies a specific niche in the VLA landscape:

```
                    Real Robot VLAs (RT-2, OpenVLA)
                         │
                         │  continuous motor commands
                         │  real-world deployment
                         │
    ┌────────────────────┼──────────────────────────┐
    │                    │                          │
    │  SimLingo          │                          │
    │  (driving)         │                          │
    │  continuous        │   VLA-AutoParts          │
    │  actions           │   (sorting)              │
    │                    │   discrete decisions +   │
    │                    │   simulated trajectories │
    │                    │                          │
    └────────────────────┼──────────────────────────┘
                         │
                    VLMs (SigLIP, PaLiGemma, InternVL2)
                    perception + language only
```

This project bridges the gap between pure VLMs and full robotic VLAs. It demonstrates the complete VLA architecture — vision encoder, language model, and action head with trajectory prediction — applied to an industrial sorting scenario, making the VLA paradigm accessible for implementation and study within constrained compute environments (8GB VRAM).

---

## 4. System Architecture

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   VLA-AutoParts Model                           │
│                                                                 │
│   ┌──────────────┐         ┌────────────────────────────────┐   │
│   │ InternViT    │         │ InternLM-2 (4B)               │   │
│   │ 300M params  │ visual  │ QLoRA adapted (r=16)          │   │
│   │ (frozen)     │─tokens─▶│                                │   │
│   │              │         │ Generates text response        │   │
│   └──────────────┘         └───────────┬────────────────────┘   │
│                                        │                        │
│        Image                    hidden states (4096-d)          │
│        Input                           │                        │
│                           ┌────────────┴────────────┐           │
│                           │ Disentangled Action Head│           │
│                           │                         │           │
│                           │  SharedProjection       │           │
│                           │  (4096 → 512)           │           │
│                           │       │                 │           │
│                ┌──────────┼───────┼────────┐        │           │
│                │          │       │        │        │           │
│                ▼          ▼       ▼        ▼        │           │
│         Classification  Sorting Action  Trajectory  │           │
│         Branch          Branch          Branch      │           │
│         (50 classes)    (bin/pri/insp)  (7×3 coords)│           │
│                           │                         │           │
│                           └─────────────────────────┘           │
│                                                                 │
│   Text Output:                  Structured Output:              │
│   "Part: brake_caliper          class=0, bin=1,                 │
│    Condition: corroded           priority=high,                 │
│    Action: bin=1, high,          inspect=true,                  │
│    Reason: safety-critical..."   trajectory=[(x,y,z)×7]        │
└─────────────────────────────────────────────────────────────────┘
```

**File:** `src/model/vla.py` — `VLAAutoPartsModel` class

The model operates in two modes:

- **Training**: Both the language head (causal LM loss on text tokens) and the action head (classification + regression losses on structured outputs) are trained simultaneously via a combined loss function.
- **Inference**: The language head generates a human-readable explanation while the action head produces machine-readable, guaranteed-valid structured outputs in a single forward pass.

### 4.2 Vision Encoder — InternViT-300M

The vision encoder is **InternViT-300M**, a Vision Transformer with 300 million parameters from the InternVL2 family. It processes 448×448 pixel images and produces a sequence of visual tokens that are projected into the language model's embedding space.

**Key design decision:** The vision encoder is **frozen** during fine-tuning. Its pre-trained visual representations are already strong enough for part recognition. Freezing it saves approximately 1.2GB of VRAM that would otherwise be consumed by optimizer states and gradients — critical for fitting within the 8GB budget.

### 4.3 Language Model — InternLM-2 (4B) with QLoRA

The language backbone is **InternLM-2** with 4 billion parameters, adapted using QLoRA:

**QLoRA** (Quantized Low-Rank Adaptation) combines two techniques:

1. **4-bit NF4 quantization** — The pre-trained weights are stored in 4-bit Normal Float format, reducing the 4B model's memory footprint from ~8GB to ~2GB. Double quantization is applied (quantizing the quantization constants) for an additional ~0.4GB saving.

2. **LoRA adapters** (rank r=16) — Instead of fine-tuning all 4B parameters, small rank-16 adapter matrices are injected into the attention and MLP layers. Only these adapters are trained, reducing trainable parameters from 4B to approximately 20M.

**File:** `src/model/vla.py` — `build_qlora_config()`, `build_lora_config()`

```python
# QLoRA configuration
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA targets: attention + MLP layers of InternLM-2
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)
```

### 4.4 Disentangled Action Head

**File:** `src/model/action_head.py`

The action head is the core contribution of this project. It takes the language model's hidden state (a 4096-dimensional vector) and produces four structured outputs through three disentangled branches.

**Parameter counts:**

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| Shared Projection | 2,098,688 | Maps 4096-d LLM hidden state → 512-d feature vector |
| Classification Branch | 144,178 | 50-class part identification |
| Sorting Action Branch | 135,183 | Bin (10), Priority (4), Inspect (1) |
| Trajectory Branch | 166,933 | 7 waypoints × 3 coordinates |
| **Total** | **2,544,982** | ~2.5M parameters (trained from scratch) |

**Architecture detail:**

```
LLM Hidden State (batch, seq_len, 4096)
    │
    ▼ [pool: extract last non-padding token]
(batch, 4096)
    │
    ▼ SharedProjection: Linear(4096→512) → LayerNorm → GELU → Dropout
(batch, 512)
    │
    ├──► ClassificationBranch:
    │       Linear(512→256) → GELU → Dropout → Linear(256→50)
    │       Output: logits_class (batch, 50) → argmax → part index
    │
    ├──► SortingActionBranch:
    │       Linear(512→256) → GELU → Dropout →
    │         ├─ Linear(256→10) → logits_bin      → argmax → bin (1-10)
    │         ├─ Linear(256→4)  → logits_priority → argmax → priority level
    │         └─ Linear(256→1)  → logit_inspect   → sigmoid → inspect flag
    │
    └──► TrajectoryBranch:
            Linear(512→256) → GELU → Dropout →
            Linear(256→128) → GELU → Dropout →
            Linear(128→21) → raw coordinates (7 waypoints × 3)
```

**Why disentangled?** Following SimLingo's finding, the classification and sorting branches share the same 512-d projection but have independent downstream networks. This prevents gradient interference: the classification loss optimizing for "what part is this?" does not distort the sorting loss optimizing for "which bin does it go to?". Empirically, disentangled architectures produce higher action accuracy than a single unified head.

### 4.5 Trajectory Prediction Branch

The trajectory branch predicts 21 continuous values representing a 7-waypoint pick-and-place trajectory:

| Waypoint | Name | Purpose |
|----------|------|---------|
| 1 | Approach | Hover above the part on the conveyor |
| 2 | Pick | Descend to grasp the part |
| 3 | Lift | Lift straight up to safe transit height |
| 4 | Move | Horizontal transit to above the target bin |
| 5 | Lower | Descend toward the bin |
| 6 | Place | Lower into the bin and release |
| 7 | Retreat | Lift away from the bin |

Each waypoint is an (x, y, z) coordinate in meters, totaling 21 output values. The branch uses MSE (Mean Squared Error) loss for continuous regression, as the output space is naturally Euclidean.

### 4.6 Dual-Loss Training Objective

The total training loss combines the language modeling loss with the structured action head loss:

```
L_total = λ_lang × L_language + λ_action × L_action

where L_action = λ_class    × CrossEntropy(logits_class, label_class)
              + λ_bin      × CrossEntropy(logits_bin, label_bin)
              + λ_priority × CrossEntropy(logits_priority, label_priority)
              + λ_inspect  × BCE(logit_inspect, label_inspect)
              + λ_traj     × MSE(pred_trajectory, label_trajectory)
```

Default loss weights:
- `λ_lang = 1.0` — standard causal language modeling loss
- `λ_action = 0.5` — overall action head weight
- `λ_class = 1.0`, `λ_bin = 1.0`, `λ_priority = 1.0` — equal classification weights
- `λ_inspect = 2.0` — **doubled** because inspection is safety-critical
- `λ_traj = 1.0` — trajectory regression

The inspection loss is weighted 2× higher because a missed inspection on a safety-critical part (e.g., a corroded brake caliper released without human review) represents the highest-risk failure mode of the system.

---

## 5. Dataset Construction

### 5.1 Part Taxonomy — 50 Classes

**File:** `src/dataset/classes.py`

The dataset covers 50 automotive part classes organized into 4 categories with 10 physical sorting bins:

| Category | Bins | Parts | Safety-Critical |
|----------|------|-------|:-:|
| Braking | 1–3 | 5 (caliper, rotor, pad set, drum, brake line) | All 5 |
| Suspension | 4–5 | 13 (wheel bearing, CV joint, tie rod, ball joint, control arm, shock, strut, spring, sway bar, hub, steering rack, PS pump, lug nuts) | 10 of 13 |
| Engine / Drivetrain | 6–7 | 29 (alternator, starter, ignition coil, spark plug, injector, fuel pump, filters, radiator, water pump, thermostat, belts, exhaust, sensors, turbo, intake, valve cover, gaskets, flywheel, clutch, transmission filter, differential cover) | 3 of 29 |
| Electrical / Accessories | 8–9 | 3 (heater core, A/C compressor, wiper motor) | 0 of 3 |
| **Reject** | **10** | Any severely damaged part regardless of category | — |

**Total: 50 classes, 17 safety-critical, 33 non-safety-critical**

Each class is defined as a `PartClass` dataclass with:
- `idx` — 0-based class index for the classification head
- `slug` — machine-readable identifier (`brake_caliper`)
- `category` — determines the bin group
- `default_bin` — bin assignment for parts in good condition
- `safety_critical` — flag that elevates handling for defective parts

### 5.2 Condition Assessment Framework

Six condition levels, ordered by severity:

| Condition | Description | Frequency in Dataset |
|-----------|-------------|:---:|
| `new` | Factory-fresh, no signs of use | 20% |
| `good` | Minimal use, well within serviceable limits | 20% |
| `minor_wear` | Normal wear from extended use, still functional | 15% |
| `minor_corrosion` | Surface oxidation, no structural compromise | 15% |
| `damaged` | Significant structural or functional compromise | 20% |
| `severely_damaged` | Catastrophic failure, unusable | 10% |

The distribution is intentionally weighted: `damaged` and `severely_damaged` cases, while less common in practice, are overrepresented to ensure the model encounters sufficient examples of the safety-critical decision boundary.

### 5.3 Deterministic Sorting Rule Engine

**File:** `src/dataset/sorting_rules.py`

The rule engine is a pure function that maps `(part_class, condition) → (bin, priority, inspect, reason)`. It serves as the **ground truth generator** for the dataset — it is not used at inference time.

**Decision logic:**

```
f(part, condition):
    if condition ∈ {new, good, minor_wear}:
        → (default_bin, normal, no_inspect)

    if condition = minor_corrosion:
        if part.safety_critical:
            → (default_bin, HIGH, INSPECT)
            reason: "corrosion may compromise structural integrity"
        else:
            → (default_bin, normal, no_inspect)
            reason: "cosmetic corrosion does not affect function"

    if condition = damaged:
        → (default_bin, HIGH, INSPECT)

    if condition = severely_damaged:
        → (BIN 10 (reject), URGENT, INSPECT)
```

The critical decision boundary is **minor corrosion on safety-critical parts**. A corroded brake caliper must be flagged for human inspection; a corroded oil filter does not. This distinction — which requires joint reasoning about the part identity and its condition — is the primary capability the model must learn.

The rule engine generates 300 unique (part, condition) combinations (50 parts × 6 conditions), each producing a deterministic sorting action. This determinism ensures perfect label consistency across the dataset.

### 5.4 Image Acquisition Pipeline

**File:** `src/dataset/image_downloader.py`

Images are acquired via web search using DuckDuckGo and Bing image APIs:

- **3 search queries per class**, crafted to return isolated product photos (e.g., `"brake caliper auto part isolated"`, `"brake caliper product photo white background"`)
- **Target: 60 images per class**, approximately 3,000 total
- **Quality gates**: minimum 150px resolution, valid JPEG/PNG, MD5-based deduplication
- **Rate limiting**: polite delays between requests, exponential backoff on rate limits

**Current dataset: ~3,250 raw images across 50 classes (~65 per class)**

### 5.5 Annotation and Conversation Format

**File:** `src/dataset/pipeline.py`

The annotation pipeline transforms raw images into InternVL2-compatible training samples:

1. **Clean** — Resize to 448×448 minimum, convert to standardized JPEG
2. **Assign conditions** — Distributed across images according to the target distribution
3. **Apply sorting rules** — Generate ground-truth bin, priority, inspect, and reason
4. **Generate trajectory** — Compute pick-and-place waypoints for the assigned bin
5. **Format as InternVL2 conversation** — Structure as human/GPT turn pairs

**Sample record format:**

```json
{
  "id": "brake_caliper_0003",
  "image": "processed/brake_caliper/brake_caliper_good_0003.jpg",
  "part_slug": "brake_caliper",
  "condition": "minor_corrosion",
  "bin": 1,
  "priority": "high",
  "inspect": true,
  "trajectory_waypoints": [[0.25, 0.54, 0.10], [0.25, 0.54, 0.01], ...],
  "pick_point": [0.25, 0.54, 0.02],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIdentify this automotive part, assess its condition, and provide the sorting action."
    },
    {
      "from": "gpt",
      "value": "Part: brake_caliper\nCondition: minor_corrosion\nConfidence: 0.95\nAction: bin=1, priority=high, inspect=true\nReason: Minor corrosion detected on Brake Caliper, which is a safety-critical component..."
    }
  ]
}
```

The `<image>` token in the human turn is a placeholder that InternVL2's processor replaces with visual features from the vision encoder.

**Manual annotation tool** (`src/dataset/annotator.py`): A Gradio web interface allows human reviewers to correct condition labels. The annotator only modifies the condition — the sorting action recomputes automatically from the rule engine, ensuring consistency.

### 5.6 Sort Dreaming — Language-Action Alignment

**File:** `src/dataset/sort_dreaming.py`

Adapted from SimLingo's **Action Dreaming**, Sort Dreaming generates 500 text-only samples that test whether the model can derive correct sorting actions from language descriptions alone, without visual input.

**Three types of samples:**

**Type 1 — Templated descriptions (bulk):**
> "A brand-new Brake Caliper still in its original manufacturer packaging. No signs of use, corrosion, or damage. Protective coating intact. What is the sorting action?"

Four templates per condition × 50 parts, randomly sampled to fill the target count.

**Type 2 — Hand-crafted edge cases (10 samples):**
> "A brake caliper with slight pitting visible on the piston bore and minor rust on the mounting bracket. No cracks are present. What is the sorting action?"

These specifically test the safety-critical decision boundary — does the model know that corrosion on a brake caliper (safety-critical) requires inspection, while corrosion on an oil filter (non-safety-critical) does not?

**Type 3 — Cross-category reasoning:**
> "A shock absorber leaking fluid from the piston rod seal. The outer body is dented near the lower mount and the rod shows scoring."

Tests compound defect assessment across part categories.

### 5.7 Trajectory Ground Truth Generation

**Files:** `src/simulation/workspace.py`, `src/simulation/trajectory.py`

The simulated workspace defines a physical environment in meters:

```
Conveyor Belt          Bins (2 columns × 5 rows)
x: [0.10, 0.35]       Column 1 (x=0.65): Bins 1-5
y: [0.15, 0.85]       Column 2 (x=0.85): Bins 6-10
z: 0.02 (surface)     z: 0.05 (bin height)
                       Safe transit height: 0.30m
```

For each training sample, the trajectory generator:
1. Determines the target bin from the sorting rules
2. Samples a random pick position on the conveyor (seeded by sample ID for reproducibility)
3. Generates 7 waypoints: approach → pick → lift → move → lower → place → retreat

Each waypoint is a 3D coordinate, producing 21 ground truth values per sample.

### 5.8 Data Splits and Validation

**Stratified split** (by part class) ensures each class appears in every split:

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (never seen during training) |

**Validation checks:**
- Rule consistency — every annotation matches the deterministic rule engine
- Image integrity — every referenced image exists and is valid
- Class balance — each class has 45–55 samples total
- Trajectory validity — waypoint count and coordinate ranges

---

## 6. Training Methodology

### 6.1 QLoRA Fine-Tuning Strategy

The training strategy is designed to adapt a large pre-trained model to the sorting domain while keeping compute requirements manageable:

| Component | Training Mode | Parameters |
|-----------|--------------|------------|
| InternViT-300M (vision) | **Frozen** | 0 trainable |
| InternLM-2 (language) | **QLoRA** (4-bit + LoRA r=16) | ~20M trainable |
| Action Head | **Full precision** (trained from scratch) | 2.5M trainable |
| **Total trainable** | | **~22.5M** (0.5% of total) |

The vision encoder is frozen because pre-trained visual features are sufficient for part recognition. The language model is adapted via QLoRA to learn the sorting-specific vocabulary and reasoning patterns. The action head is trained from scratch in full precision because its weights are initialized randomly and must learn entirely new mappings.

### 6.2 Memory-Efficient Training for 8GB VRAM

**File:** `src/training/trainer.py`

| Technique | VRAM Saved | Description |
|-----------|-----------|-------------|
| 4-bit NF4 quantization | ~6GB | Reduces 4B model from ~8GB to ~2GB |
| Double quantization | ~0.4GB | Quantizes quantization constants |
| Frozen vision encoder | ~1.2GB | No optimizer states for 300M params |
| Gradient checkpointing | ~1GB | Recomputes activations during backward pass |
| Batch size 1 + grad accumulation 8 | ~0.5GB | Effective batch of 8 without storing 8 samples |
| bfloat16 compute | — | Reduces intermediate tensor precision |

Total VRAM footprint: approximately 5–6GB, fitting comfortably within an RTX 4060 (8GB).

### 6.3 Multi-Task Loss Function

The model is trained on two simultaneous objectives:

**Language loss** — Standard causal language modeling. The model generates the GPT response token by token, and cross-entropy loss is computed only on the response tokens (the human turn is masked with label=-100).

**Action loss** — The combined loss from all action head branches, as detailed in Section 4.6.

```
L_total = 1.0 × L_language + 0.5 × L_action
```

The 0.5 weight on the action loss ensures that language generation quality is not sacrificed for action accuracy. The language response serves as a human-readable explanation, while the action head provides the machine-readable decision.

### 6.4 Training Configuration

**File:** `src/training/trainer.py` — `default_config()`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Sufficient for convergence with QLoRA |
| Learning rate | 2e-4 | Standard for QLoRA fine-tuning |
| Scheduler | OneCycleLR (cosine) | Warmup + cosine decay |
| Warmup | 5% of total steps | Prevents early instability |
| Weight decay | 0.01 | Mild regularization |
| Max gradient norm | 1.0 | Gradient clipping for stability |
| Batch size | 1 | VRAM constraint |
| Gradient accumulation | 8 | Effective batch = 8 |
| Max sequence length | 512 tokens | Covers full conversation |
| Precision | bfloat16 | Optimal for modern GPUs |

**Optimizer:** AdamW with weight decay 0.01 on all trainable parameters (LoRA adapters + action head).

**Checkpointing:** LoRA adapters and action head weights are saved separately, enabling independent loading and evaluation.

---

## 7. Evaluation Framework

**File:** `src/eval/benchmark.py`

### 7.1 Action Head Metrics

Evaluated on the held-out test set using the action head's structured outputs:

| Metric | Description | Target |
|--------|-------------|--------|
| **Class Accuracy** | Correct part identification out of 50 classes | >85% |
| **Bin Accuracy** | Correct bin assignment out of 10 bins | >90% |
| **Priority Accuracy** | Correct priority level out of 4 levels | >90% |
| **Inspect F1** | F1 score for binary inspection flag | >85% |
| **Inspect Precision** | Of parts flagged for inspection, how many truly need it | High |
| **Inspect Recall** | Of parts that need inspection, how many are caught | Critical — must be >95% |

**Inspect recall is the most safety-critical metric.** A false negative (failing to flag a damaged brake caliper for inspection) is far more dangerous than a false positive (unnecessarily flagging a healthy part). The 2× loss weight on the inspect branch reflects this priority.

### 7.2 Sort Dreaming Evaluation

The 500 Sort Dreaming samples are evaluated by:
1. Feeding the text description (no image) into the model
2. Generating a text response
3. Parsing the response with regex to extract predicted part, bin, priority, and inspect
4. Comparing against the deterministic ground truth

This measures **language-to-action alignment** — can the model derive the correct sorting action from a textual description alone? High Sort Dreaming accuracy indicates that the model has internalized the sorting rules, not just memorized image-action pairs.

### 7.3 Zero-Shot Baseline Comparison

A stock InternVL2 model (no fine-tuning, no LoRA, no action head) is evaluated on the same tasks using text generation only. This baseline quantifies the improvement gained from fine-tuning:

- The baseline model has no knowledge of the sorting rules
- Its responses must be parsed from free-form text
- It may hallucinate bins, priorities, or inspection flags that don't exist

The expected result: low accuracy across all metrics, demonstrating that domain-specific fine-tuning is necessary.

### 7.4 Trajectory Evaluation Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Mean Waypoint Error (MWE)** | Average L2 distance across all 7 waypoints | meters |
| **Final Placement Error (FPE)** | L2 distance at the "place" waypoint (index 5) | meters |
| **Success Rate** | Fraction where FPE < 2cm threshold | percentage |

The placement error at waypoint 6 (place) is the most critical — it determines whether the part lands in the correct bin. A robot arm with 2cm placement accuracy would reliably sort parts into standard bin openings.

---

## 8. Simulated Pick-and-Place Environment

### 8.1 Workspace Layout

**File:** `src/simulation/workspace.py`

The simulated workspace represents a 1m × 1m physical area:

```
   Y (meters)
   0.9 ┌──────────┐      ┌─────┐  ┌─────┐
       │ Conveyor │      │ B1  │  │ B6  │
   0.7 │ (pickup  │      │Brake│  │Eng. │
       │  zone)   │      ├─────┤  ├─────┤
   0.5 │          │      │ B3  │  │ B8  │
       │          │      ├─────┤  ├─────┤
   0.3 │          │      │ B4  │  │ B9  │
       │          │      ├─────┤  ├─────┤
   0.1 └──────────┘      │ B5  │  │ B10 │
                          └─────┘  └─────┘
       0.0  0.1  0.35    0.65    0.85    1.0 → X (meters)
```

- **Conveyor zone**: x ∈ [0.10, 0.35], y ∈ [0.15, 0.85] — parts arrive at random positions
- **Bins**: Two columns of 5, spaced at x=0.65 and x=0.85
- **Safe transit height**: z=0.30m — robot arm clears all obstacles during horizontal moves
- **Bin height**: z=0.05m — slightly raised bin walls

### 8.2 Trajectory Generation

**File:** `src/simulation/trajectory.py`

Each trajectory is a deterministic function of the pick point and target bin:

```
trajectory = f(pick_point, target_bin)

where:
  pick_point = random position on conveyor (seeded by sample ID)
  target_bin = output of sorting rules (deterministic)
```

The 7 waypoints are computed geometrically:

| # | Name | x | y | z |
|---|------|---|---|---|
| 1 | Approach | pick_x | pick_y | pick_z + 0.08 |
| 2 | Pick | pick_x | pick_y | pick_z - 0.01 |
| 3 | Lift | pick_x | pick_y | 0.30 (safe height) |
| 4 | Move | bin_x | bin_y | 0.30 (safe height) |
| 5 | Lower | bin_x | bin_y | bin_z + 0.08 |
| 6 | Place | bin_x | bin_y | bin_z |
| 7 | Retreat | bin_x | bin_y | 0.40 |

**Reproducibility:** Each sample's pick position is seeded by an MD5 hash of the sample ID. Running the pipeline with the same seed produces identical trajectories, while different parts land at different conveyor positions, adding natural variation.

### 8.3 Visualization

**File:** `src/simulation/visualize.py`

Three visualization modes are provided:

1. **2D Top-Down View** — Shows the workspace from above with the conveyor zone, all 10 bins (color-coded by category), and the trajectory path with labeled waypoints.

2. **3D View** — Full 3D rendering showing the height profile of the trajectory — the lift, transit, and lowering phases are clearly visible in the Z dimension.

3. **Comparison View** — Overlays predicted and ground truth trajectories, annotating the error at each waypoint in meters. Used for evaluating model performance.

---

## 9. Future Enhancements

### 9.1 Visual Condition Assessment

The current dataset assigns conditions randomly to images — a clean image may be labeled "damaged." Future work should address this through:

- **Condition-specific image acquisition** — Search queries targeting "damaged brake caliper cracked", "corroded brake caliper rust", etc.
- **Synthetic damage augmentation** — Algorithmically applying rust textures, crack patterns, warping distortions, and color degradation to clean images, creating visually accurate training pairs.
- **Foundation model annotation** — Using GPT-4V or similar models to assess and label the actual condition of each image before training.

This would enable the model to learn true visual condition assessment — distinguishing a damaged part from a new one by appearance alone.

### 9.2 Real Robotic Integration

The current trajectory prediction is simulated. Transitioning to a real robotic system would require:

- **Closed-loop control** — Predicting one waypoint at a time, observing the result, and correcting. The current open-loop 7-waypoint prediction assumes perfect execution.
- **Depth estimation** — Adding a depth sensor or stereo camera for accurate Z-coordinate prediction.
- **Gripper force prediction** — Different parts require different grasping forces. A delicate sensor requires gentle handling; a heavy flywheel requires firm grip.
- **Collision avoidance** — Ensuring the predicted trajectory does not pass through obstacles or other bins.

### 9.3 Multi-View and Video Input

Real sorting lines may benefit from:

- **Multi-view fusion** — Multiple cameras at different angles provide more complete part geometry information, improving classification accuracy for parts that look similar from one angle.
- **Video input** — Processing a sequence of frames as the part moves on the conveyor, rather than a single snapshot. This would enable temporal reasoning (e.g., detecting rattling inside a catalytic converter from vibration patterns).

### 9.4 Continual Learning

Deployed systems encounter new part types and novel failure modes:

- **Online adaptation** — Updating the model with new examples without full retraining.
- **Active learning** — The model flags low-confidence predictions for human review, and those corrections are fed back into the training set.
- **Domain shift detection** — Alerting operators when incoming parts differ significantly from the training distribution (e.g., parts from a new vehicle manufacturer).

### 9.5 Safety and Compliance

For production deployment:

- **Uncertainty quantification** — The action head currently outputs point predictions. Adding confidence calibration (e.g., temperature scaling on logits) would enable the system to abstain when uncertain.
- **Audit trail** — Logging every sorting decision with the model's reasoning, the image, and the predicted trajectory for regulatory compliance.
- **Human-in-the-loop** — The `inspect=true` flag already implements this partially. Extending it to a confidence threshold (inspect if confidence < 0.8) would provide a tunable safety margin.

---

## 10. Conclusion

VLA-AutoParts demonstrates the practical application of the Vision-Language-Action paradigm to industrial automation. By extending a pre-trained Vision-Language Model (InternVL2) with a custom disentangled action head inspired by SimLingo, the system achieves the complete perception-reasoning-action loop required for autonomous parts sorting:

1. **Vision** — InternViT-300M identifies the part from a camera image.
2. **Language** — InternLM-2 generates a human-readable assessment and sorting rationale.
3. **Action** — The disentangled action head produces machine-readable sorting decisions (bin, priority, inspection flag) and pick-and-place trajectory predictions.

The project makes several contributions:

- **A 50-class automotive parts taxonomy** with safety-critical annotations, condition assessment framework, and deterministic sorting rules.
- **A disentangled action head architecture** with separate classification, sorting, and trajectory branches, adapted from SimLingo's design for the sorting domain.
- **Sort Dreaming evaluation protocol** — adapted from SimLingo's Action Dreaming — testing language-to-action alignment without visual input.
- **Memory-efficient training pipeline** — QLoRA fine-tuning with 4-bit quantization, gradient checkpointing, and frozen vision encoder, fitting within an 8GB VRAM budget.
- **A simulated pick-and-place environment** with workspace layout, trajectory generation, and visualization tools.

The architecture is designed for extensibility. The modular separation between the sorting rule engine (ground truth), the model (learned approximation), and the simulation (trajectory generation) allows each component to be improved independently. The path from the current simulation to real-world robotic deployment is a matter of replacing simulated trajectories with teleoperated demonstrations and adding closed-loop control — the model architecture itself requires no modification.

VLA-AutoParts bridges the gap between academic VLM research and industrial robotic automation, demonstrating that the VLA paradigm — which has shown remarkable results in autonomous driving (SimLingo) and general-purpose robotics (RT-2, OpenVLA) — can be effectively adapted for manufacturing and logistics applications within practical compute constraints.

---

## 11. References

1. **SimLingo** — Renz, K., et al. "SimLingo: Vision-Language-Action Model for Autonomous Driving with Disentangled Action Head." CVPR 2025.

2. **InternVL2** — Chen, Z., et al. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." CVPR 2024.

3. **RT-2** — Brohan, A., et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." CoRL 2023.

4. **OpenVLA** — Kim, M., et al. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv 2024.

5. **QLoRA** — Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.

6. **LoRA** — Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

7. **SigLIP** — Zhai, X., et al. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023.

8. **PaLiGemma** — Beyer, L., et al. "PaLiGemma: A Versatile 3B VLM for Transfer." arXiv 2024.

9. **CLIP** — Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.

10. **BitsAndBytes** — Dettmers, T., et al. "8-bit Optimizers via Block-wise Quantization." ICLR 2022.

---
