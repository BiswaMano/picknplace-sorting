"""
Deterministic sorting-rule engine for VLA-AutoParts.

Given a part class and a condition label, this module produces the
canonical (bin, priority, inspect, reason) annotation.  These rules
serve as ground truth for dataset construction and are also used at
inference time to validate model outputs.

Rule logic (matches PDF Section 4.3):
  new / good          → default_bin, normal, no-inspect
  minor_wear          → default_bin, normal, no-inspect
  minor_corrosion     → safety-critical?  high+inspect : normal+no-inspect
  damaged             → default_bin, high, inspect
  severely_damaged    → bin 10 (reject), urgent, inspect
"""

from dataclasses import dataclass

from src.dataset.classes import (
    CONDITIONS,
    PART_CLASSES,
    SLUG_TO_CLASS,
    IDX_TO_CLASS,
    PartClass,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class SortingAction:
    part_slug: str
    condition: str
    bin: int
    priority: str       # low | normal | high | urgent
    inspect: bool
    reason: str

    def to_response_text(self, confidence: float = 0.95) -> str:
        """Format as the GPT response string used in InternVL2 conversations."""
        inspect_str = str(self.inspect).lower()
        return (
            f"Part: {self.part_slug}\n"
            f"Condition: {self.condition}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Action: bin={self.bin}, priority={self.priority}, inspect={inspect_str}\n"
            f"Reason: {self.reason}"
        )


# ---------------------------------------------------------------------------
# Reason templates
# ---------------------------------------------------------------------------
def _reason(part: PartClass, condition: str, bin_: int, priority: str, inspect: bool) -> str:
    """Generate a deterministic, natural-language reason string."""
    bin_group = _bin_label(bin_)

    if condition in ("new", "good"):
        return (
            f"{part.name} in {condition} condition with no visible defects. "
            f"Standard routing to {bin_group}."
        )
    if condition == "minor_wear":
        return (
            f"{part.name} shows minor wear consistent with normal use. "
            f"Still serviceable; routed to {bin_group}."
        )
    if condition == "minor_corrosion":
        if part.safety_critical:
            return (
                f"Minor corrosion detected on {part.name}, which is a safety-critical component. "
                f"Corrosion may compromise structural integrity or sealing surfaces. "
                f"Routed to {bin_group} with elevated priority; human inspection required."
            )
        return (
            f"Minor corrosion on {part.name}. Component is not safety-critical; "
            f"cosmetic corrosion does not affect function. Routed to {bin_group}."
        )
    if condition == "damaged":
        return (
            f"{part.name} shows significant damage. Functional integrity is uncertain. "
            f"Routed to {bin_group} with high priority; human inspection required before disposition."
        )
    if condition == "severely_damaged":
        return (
            f"{part.name} is severely damaged and cannot be safely refurbished or resold. "
            f"Routed to Bin 10 (reject / inspection hold) for disposal or detailed assessment."
        )
    return f"{part.name} — condition '{condition}' routing to {bin_group}."


def _bin_label(bin_: int) -> str:
    labels = {
        1: "Bin 1 (braking — premium)",
        2: "Bin 2 (braking — standard)",
        3: "Bin 3 (braking — line components)",
        4: "Bin 4 (suspension — safety components)",
        5: "Bin 5 (suspension — ride components)",
        6: "Bin 6 (engine — sensors / ignition)",
        7: "Bin 7 (engine — mechanical)",
        8: "Bin 8 (electrical / climate)",
        9: "Bin 9 (accessories / motors)",
        10: "Bin 10 (reject / inspection hold)",
    }
    return labels.get(bin_, f"Bin {bin_}")


# ---------------------------------------------------------------------------
# Core rule engine
# ---------------------------------------------------------------------------
def apply_sorting_rules(part: PartClass, condition: str) -> SortingAction:
    """
    Apply deterministic sorting rules and return a SortingAction.

    Args:
        part:      PartClass instance from classes.py
        condition: one of CONDITIONS

    Returns:
        SortingAction with all fields populated
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Must be one of {CONDITIONS}")

    # Defaults
    bin_ = part.default_bin
    priority = "normal"
    inspect = False

    if condition in ("new", "good"):
        pass  # defaults hold

    elif condition == "minor_wear":
        pass  # defaults hold

    elif condition == "minor_corrosion":
        if part.safety_critical:
            priority = "high"
            inspect = True
        # else defaults hold

    elif condition == "damaged":
        priority = "high"
        inspect = True

    elif condition == "severely_damaged":
        bin_ = 10
        priority = "urgent"
        inspect = True

    reason = _reason(part, condition, bin_, priority, inspect)
    return SortingAction(
        part_slug=part.slug,
        condition=condition,
        bin=bin_,
        priority=priority,
        inspect=inspect,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def sort_by_slug(slug: str, condition: str) -> SortingAction:
    part = SLUG_TO_CLASS.get(slug)
    if part is None:
        raise ValueError(f"Unknown part slug '{slug}'")
    return apply_sorting_rules(part, condition)


def sort_by_idx(idx: int, condition: str) -> SortingAction:
    part = IDX_TO_CLASS.get(idx)
    if part is None:
        raise ValueError(f"Unknown part index {idx}")
    return apply_sorting_rules(part, condition)


# ---------------------------------------------------------------------------
# Validation helper — checks consistency of a full annotation dict
# ---------------------------------------------------------------------------
def validate_annotation(ann: dict) -> list[str]:
    """
    Returns a list of error strings.  Empty list means annotation is consistent.
    """
    errors: list[str] = []
    slug = ann.get("part_slug") or ann.get("part_class")
    condition = ann.get("condition")

    if slug not in SLUG_TO_CLASS:
        errors.append(f"Unknown slug: {slug}")
        return errors
    if condition not in CONDITIONS:
        errors.append(f"Unknown condition: {condition}")
        return errors

    expected = apply_sorting_rules(SLUG_TO_CLASS[slug], condition)
    if ann.get("bin") != expected.bin:
        errors.append(f"bin mismatch: got {ann.get('bin')}, expected {expected.bin}")
    if ann.get("priority") != expected.priority:
        errors.append(f"priority mismatch: got {ann.get('priority')}, expected {expected.priority}")
    if ann.get("inspect") != expected.inspect:
        errors.append(f"inspect mismatch: got {ann.get('inspect')}, expected {expected.inspect}")
    return errors


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ("brake_caliper",    "new"),
        ("brake_caliper",    "minor_corrosion"),
        ("brake_caliper",    "severely_damaged"),
        ("oil_filter",       "minor_corrosion"),
        ("shock_absorber",   "damaged"),
        ("timing_belt",      "minor_wear"),
        ("ac_compressor",    "good"),
    ]
    for slug, cond in test_cases:
        action = sort_by_slug(slug, cond)
        print(f"\n[{slug}] condition={cond}")
        print(action.to_response_text())
        print("-" * 60)
