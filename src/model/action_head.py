"""
Disentangled Action Head for VLA-AutoParts.

Architecture (mirrors SimLingo's disentangled action head design):

  LLM hidden state (d_model)
       │
       ▼
  Shared projection (d_model → d_hidden)
       │
  ┌────┴────────────────────────────┐
  │                                 │
  ▼                                 ▼
Classification Branch        Sorting-Action Branch
(50-class part recognition)  (bin + priority + inspect)
  │                                 │
  └─── logits_class (50)     ┌──────┼────────┐
                             │      │        │
                           bin   priority  inspect
                          (10)    (4)       (1)

The two branches are kept separate (disentangled) so that the
classification signal does not bleed into the action regression
and vice versa — following SimLingo's finding that disentanglement
improves action accuracy.

The head is appended to the frozen/LoRA-adapted LLM and trained
end-to-end during QLoRA fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.classes import (
    NUM_CLASSES,    # 50
    NUM_BINS,       # 10
    PRIORITIES,     # 4 values
    CONDITIONS,     # 6 values
)


NUM_PRIORITIES = len(PRIORITIES)    # 4
NUM_CONDITIONS = len(CONDITIONS)    # 6


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SharedProjection(nn.Module):
    """Maps LLM hidden state → shared feature vector."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationBranch(nn.Module):
    """50-class part recognition branch."""

    def __init__(self, d_hidden: int, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, num_classes)."""
        return self.net(x)


class SortingActionBranch(nn.Module):
    """
    Sorting action branch — predicts bin, priority, and inspect flag.

    Outputs:
      logits_bin      (batch, NUM_BINS)      — 10-class bin classification
      logits_priority (batch, NUM_PRIORITIES) — 4-class priority classification
      logit_inspect   (batch, 1)             — binary inspect flag (sigmoid)
    """

    def __init__(self, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        d_branch = d_hidden // 2

        # Shared within sorting branch
        self.shared = nn.Sequential(
            nn.Linear(d_hidden, d_branch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bin_head      = nn.Linear(d_branch, NUM_BINS)
        self.priority_head = nn.Linear(d_branch, NUM_PRIORITIES)
        self.inspect_head  = nn.Linear(d_branch, 1)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        return (
            self.bin_head(feat),       # (B, 10)
            self.priority_head(feat),  # (B, 4)
            self.inspect_head(feat),   # (B, 1)
        )


class TrajectoryBranch(nn.Module):
    """
    Trajectory prediction branch — predicts 7 waypoints x 3 coordinates = 21 values.

    Predicts the pick-and-place trajectory that a robot arm would follow
    to sort the part into its assigned bin.

    Outputs:
      pred_trajectory  (batch, num_waypoints * 3) — raw coordinate predictions
    """

    NUM_WAYPOINTS = 7
    COORDS_PER_WP = 3

    def __init__(self, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        d_branch = d_hidden // 2
        d_mid = d_hidden // 4
        out_dim = self.NUM_WAYPOINTS * self.COORDS_PER_WP   # 21

        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_branch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_branch, d_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 21) — raw coordinate predictions."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Full disentangled action head
# ---------------------------------------------------------------------------

class DisentangledActionHead(nn.Module):
    """
    Full disentangled action head.

    Args:
        d_model:   Hidden dimension of the LLM backbone (e.g. 4096 for 4B model).
        d_hidden:  Projection dimension (default 512 — keeps head lightweight
                   for 8GB VRAM budget).
        dropout:   Dropout rate applied throughout.

    Forward:
        x: Tensor of shape (batch, seq_len, d_model) — LLM output hidden states.
           We pool over the sequence dimension (last token = EOS).

    Returns:
        ActionHeadOutput namedtuple.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_hidden: int = 512,
        dropout: float = 0.1,
        enable_trajectory: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.enable_trajectory = enable_trajectory

        self.shared_proj     = SharedProjection(d_model, d_hidden, dropout)
        self.class_branch    = ClassificationBranch(d_hidden, NUM_CLASSES, dropout)
        self.sorting_branch  = SortingActionBranch(d_hidden, dropout)

        if enable_trajectory:
            self.trajectory_branch = TrajectoryBranch(d_hidden, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> "ActionHeadOutput":
        """
        Args:
            hidden_states:   (batch, seq_len, d_model)
            attention_mask:  (batch, seq_len) — 1 for real tokens, 0 for padding.
                             If None, uses the last token.
        Returns:
            ActionHeadOutput
        """
        # Pool: use the last non-padding token (EOS) representation
        pooled = self._pool(hidden_states, attention_mask)   # (B, d_model)

        shared = self.shared_proj(pooled)                    # (B, d_hidden)

        logits_class = self.class_branch(shared)             # (B, 50)
        logits_bin, logits_priority, logit_inspect = self.sorting_branch(shared)

        pred_trajectory = None
        if self.enable_trajectory:
            pred_trajectory = self.trajectory_branch(shared) # (B, 21)

        return ActionHeadOutput(
            logits_class=logits_class,
            logits_bin=logits_bin,
            logits_priority=logits_priority,
            logit_inspect=logit_inspect,
            pred_trajectory=pred_trajectory,
        )

    @staticmethod
    def _pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Extract the last real token representation."""
        if attention_mask is None:
            return hidden_states[:, -1, :]

        # Find the index of the last 1 in each row
        lengths = attention_mask.sum(dim=1) - 1          # (B,)
        batch_idx = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        return hidden_states[batch_idx, lengths]          # (B, d_model)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class ActionHeadOutput:
    __slots__ = (
        "logits_class", "logits_bin", "logits_priority", "logit_inspect",
        "pred_trajectory",
    )

    def __init__(
        self,
        logits_class:    torch.Tensor,
        logits_bin:      torch.Tensor,
        logits_priority: torch.Tensor,
        logit_inspect:   torch.Tensor,
        pred_trajectory: torch.Tensor | None = None,
    ):
        self.logits_class    = logits_class
        self.logits_bin      = logits_bin
        self.logits_priority = logits_priority
        self.logit_inspect   = logit_inspect
        self.pred_trajectory = pred_trajectory   # (B, 21) or None

    # ── Predictions ─────────────────────────────────────────────────────────

    @property
    def pred_class(self) -> torch.Tensor:
        return self.logits_class.argmax(dim=-1)

    @property
    def pred_bin(self) -> torch.Tensor:
        return self.logits_bin.argmax(dim=-1) + 1   # 1-indexed

    @property
    def pred_priority(self) -> torch.Tensor:
        return self.logits_priority.argmax(dim=-1)

    @property
    def pred_inspect(self) -> torch.Tensor:
        return (self.logit_inspect.squeeze(-1).sigmoid() > 0.5)

    @property
    def pred_waypoints(self) -> torch.Tensor | None:
        """Reshape trajectory (B, 21) → (B, 7, 3) for readability."""
        if self.pred_trajectory is None:
            return None
        return self.pred_trajectory.view(-1, 7, 3)

    def to_dict(self) -> dict:
        d = {
            "pred_class":    self.pred_class.tolist(),
            "pred_bin":      self.pred_bin.tolist(),
            "pred_priority": self.pred_priority.tolist(),
            "pred_inspect":  self.pred_inspect.tolist(),
        }
        if self.pred_trajectory is not None:
            d["pred_waypoints"] = self.pred_waypoints.tolist()
        return d


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class ActionHeadLoss(nn.Module):
    """
    Combined loss for the disentangled action head.

    L_total = λ_class * L_class
            + λ_bin   * L_bin
            + λ_pri   * L_priority
            + λ_ins   * L_inspect

    Weights are tunable but default to equal contribution,
    with inspect weighted slightly higher because it is safety-critical.
    """

    def __init__(
        self,
        lambda_class:      float = 1.0,
        lambda_bin:        float = 1.0,
        lambda_priority:   float = 1.0,
        lambda_inspect:    float = 2.0,   # higher weight: safety-critical signal
        lambda_trajectory: float = 1.0,
    ):
        super().__init__()
        self.lambda_class      = lambda_class
        self.lambda_bin        = lambda_bin
        self.lambda_priority   = lambda_priority
        self.lambda_inspect    = lambda_inspect
        self.lambda_trajectory = lambda_trajectory

        self.ce  = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: ActionHeadOutput,
        labels_class:      torch.Tensor,              # (B,) long
        labels_bin:        torch.Tensor,              # (B,) long, 0-indexed (bin-1)
        labels_priority:   torch.Tensor,              # (B,) long
        labels_inspect:    torch.Tensor,              # (B,) float {0,1}
        labels_trajectory: torch.Tensor | None = None,  # (B, 21) float
        has_trajectory:    torch.Tensor | None = None,  # (B,) float mask
    ) -> dict[str, torch.Tensor]:

        l_class = self.ce(output.logits_class,    labels_class)
        l_bin   = self.ce(output.logits_bin,      labels_bin)
        l_pri   = self.ce(output.logits_priority, labels_priority)
        l_ins   = self.bce(
            output.logit_inspect.squeeze(-1),
            labels_inspect.float()
        )

        total = (
            self.lambda_class    * l_class +
            self.lambda_bin      * l_bin   +
            self.lambda_priority * l_pri   +
            self.lambda_inspect  * l_ins
        )

        result = {
            "loss_class":    l_class,
            "loss_bin":      l_bin,
            "loss_priority": l_pri,
            "loss_inspect":  l_ins,
        }

        # Trajectory loss (MSE on waypoint coordinates)
        if (
            output.pred_trajectory is not None
            and labels_trajectory is not None
        ):
            if has_trajectory is not None and has_trajectory.sum() > 0:
                # Only compute loss on samples that have trajectory labels
                mask = has_trajectory.bool()
                l_traj = self.mse(
                    output.pred_trajectory[mask],
                    labels_trajectory[mask],
                )
            elif has_trajectory is None:
                l_traj = self.mse(output.pred_trajectory, labels_trajectory)
            else:
                l_traj = torch.tensor(0.0, device=total.device)

            total = total + self.lambda_trajectory * l_traj
            result["loss_trajectory"] = l_traj

        result["loss"] = total
        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    B, SEQ, D = 2, 64, 4096
    head = DisentangledActionHead(d_model=D, d_hidden=512, enable_trajectory=True)
    print(f"Action head parameters: {sum(p.numel() for p in head.parameters()):,}")

    hidden = torch.randn(B, SEQ, D)
    mask = torch.ones(B, SEQ, dtype=torch.long)
    mask[0, 50:] = 0   # simulate padding

    out = head(hidden, mask)
    print(f"logits_class:    {out.logits_class.shape}")
    print(f"logits_bin:      {out.logits_bin.shape}")
    print(f"logits_priority: {out.logits_priority.shape}")
    print(f"logit_inspect:   {out.logit_inspect.shape}")
    print(f"pred_trajectory: {out.pred_trajectory.shape}")
    print(f"pred_waypoints:  {out.pred_waypoints.shape}")
    print(f"pred_class:      {out.pred_class}")
    print(f"pred_bin:        {out.pred_bin}")
    print(f"pred_inspect:    {out.pred_inspect}")

    # Loss
    loss_fn = ActionHeadLoss()
    labels = {
        "labels_class":      torch.randint(0, NUM_CLASSES, (B,)),
        "labels_bin":        torch.randint(0, NUM_BINS, (B,)),
        "labels_priority":   torch.randint(0, NUM_PRIORITIES, (B,)),
        "labels_inspect":    torch.randint(0, 2, (B,)).float(),
        "labels_trajectory": torch.rand(B, 21),
        "has_trajectory":    torch.ones(B),
    }
    losses = loss_fn(out, **labels)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
