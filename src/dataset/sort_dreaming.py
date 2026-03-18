"""
Sort Dreaming dataset generator — adapted from SimLingo's Action Dreaming.

Generates 500 text-only hypothetical part descriptions with corresponding
action annotations.  The model must predict the correct sorting action from
language alone (no image), testing language-action alignment.

Categories of descriptions:
  1. Clear-cut cases (e.g., brand new brake caliper)
  2. Edge cases (e.g., minor surface rust on non-critical part)
  3. Ambiguous cases (borderline condition descriptions)
  4. Cross-category reasoning (comparing similar defects across part types)
  5. Compound defects (multiple issues in one description)

Usage:
  python -m src.dataset.sort_dreaming --out-file data/annotations/sort_dreaming.jsonl
  python -m src.dataset.sort_dreaming --count 500 --seed 42
"""

import random
from pathlib import Path

import jsonlines
import typer
from rich.console import Console

from src.dataset.classes import PART_CLASSES, SLUG_TO_CLASS, CONDITIONS
from src.dataset.sorting_rules import apply_sorting_rules

app = typer.Typer()
console = Console()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "annotations" / "sort_dreaming.jsonl"

# ---------------------------------------------------------------------------
# Description templates per condition
# Each template is a callable (part_name) -> description string
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list] = {
    "new": [
        lambda n: f"A brand-new {n} still in its original manufacturer packaging. No signs of use, corrosion, or damage. Protective coating intact.",
        lambda n: f"An unused {n} with factory tags attached. Surface finish is pristine and all mating surfaces are clean.",
        lambda n: f"A freshly manufactured {n} with zero hours of use. No scratches, pitting, or discoloration visible anywhere.",
        lambda n: f"A new old stock (NOS) {n} in unopened box. Part appears identical to OEM specification with no deterioration.",
    ],
    "good": [
        lambda n: f"A {n} showing minimal signs of use. Surfaces are clean, no cracks or deep scratches. Well within serviceable limits.",
        lambda n: f"A used {n} in good working condition. Light surface scuffs from handling but no structural defects or wear beyond specifications.",
        lambda n: f"A {n} that has been cleaned and inspected. Minor cosmetic marks only; all functional surfaces are undamaged.",
        lambda n: f"A low-mileage {n} removed during a routine service. No abnormal wear patterns, no leaks, no cracks.",
    ],
    "minor_wear": [
        lambda n: f"A {n} showing normal wear from extended use. Surfaces show light scoring and minor material loss, but part remains functional.",
        lambda n: f"A {n} with visible wear on contact surfaces. Within acceptable tolerance limits for continued use but approaching service interval.",
        lambda n: f"A {n} exhibiting light abrasion on working surfaces. No cracks or deep grooves; wear is evenly distributed.",
        lambda n: f"A {n} with typical high-mileage wear. Slight dimensional changes at wear points; no catastrophic degradation.",
    ],
    "minor_corrosion": [
        lambda n: f"A {n} with light surface rust on non-sealing areas. No pitting on functional surfaces; corrosion appears superficial.",
        lambda n: f"A {n} showing minor corrosion on the outer casing. Mounting threads are clean; internal components appear unaffected.",
        lambda n: f"A {n} with patchy oxidation on the body. Slight discoloration and surface roughness but no deep pitting or material loss.",
        lambda n: f"A {n} exhibiting mild corrosion typical of humid storage. Surface rust on non-critical areas only; functional geometry intact.",
    ],
    "damaged": [
        lambda n: f"A {n} with visible cracks on a primary structural surface. The part has been subjected to impact or overload stress.",
        lambda n: f"A {n} showing significant deformation. One mounting flange is bent and a sealing surface is compromised.",
        lambda n: f"A {n} with a broken retaining tab and deep gouges on the working surface. Likely non-functional in current state.",
        lambda n: f"A {n} that shows signs of overheating — discoloration, warping, and micro-cracking visible. Structural integrity is uncertain.",
    ],
    "severely_damaged": [
        lambda n: f"A {n} that is fractured in two. The failure appears catastrophic; the part is completely unusable.",
        lambda n: f"A {n} with massive impact damage — one end is sheared off and the body is severely distorted. Cannot be refurbished.",
        lambda n: f"A {n} exhibiting severe corrosion through-and-through. Material loss has created holes and the part has no structural integrity.",
        lambda n: f"A heavily corroded and cracked {n}. Multiple failure modes present simultaneously; immediate rejection required.",
    ],
}

# ---------------------------------------------------------------------------
# Edge / ambiguous case templates
# These are condition-specific but require cross-category reasoning
# ---------------------------------------------------------------------------

EDGE_CASES: list[tuple[str, str, str]] = [
    # (slug, condition, custom_description)

    # Safety-critical with minor corrosion → must inspect
    ("brake_caliper", "minor_corrosion",
     "A brake caliper with slight pitting visible on the piston bore and minor rust on the mounting bracket. "
     "No cracks are present. What is the sorting action?"),

    ("tie_rod_end", "minor_corrosion",
     "A tie rod end showing light surface oxidation near the ball stud. The boot is intact and the joint has "
     "no play when tested. Minor rust spots on the shaft only. What is the sorting action?"),

    # Non-critical with minor corrosion → normal, no inspect
    ("oil_filter", "minor_corrosion",
     "An oil filter canister with light external rust on the seam. The threads are clean and the gasket "
     "surface shows no pitting. Filter has not been used. What is the sorting action?"),

    ("muffler", "minor_corrosion",
     "A muffler with surface rust on the heat shield and inlet pipe. No holes or through-corrosion. "
     "Moderate cosmetic deterioration consistent with age. What is the sorting action?"),

    # Timing belt — safety-critical, minor wear
    ("timing_belt", "minor_wear",
     "A serpentine-style timing belt with visible glazing on the rib surfaces and minor fraying at one edge. "
     "No missing teeth or cracks. Still within manufacturer replacement interval. What is the sorting action?"),

    # Severely damaged safety-critical
    ("head_gasket", "severely_damaged",
     "A head gasket showing a clear blow-out between two cylinders. Carbon tracking visible across the fire "
     "ring. The sealing layers are delaminated at the failure point. What is the sorting action?"),

    # Ambiguous — damaged vs minor wear
    ("shock_absorber", "damaged",
     "A shock absorber leaking fluid from the piston rod seal. The outer body is dented near the lower mount "
     "and the rod shows scoring. Dampening is severely compromised. What is the sorting action?"),

    # New safety-critical part — confirm normal routing
    ("brake_rotor", "new",
     "A brand-new brake rotor still wrapped in its protective paper packaging. "
     "No signs of corrosion, machining defects, or damage. What is the sorting action?"),

    # Compound defect
    ("alternator", "damaged",
     "An alternator with a cracked housing near the front bearing and visible burn marks on the stator "
     "windings. The pulley spins but with audible roughness. Multiple failure indicators present. "
     "What is the sorting action?"),

    ("catalytic_converter", "severely_damaged",
     "A catalytic converter that has been struck by road debris — the substrate inside is shattered "
     "and the outer shell is dented through. Rattling is audible when shaken. What is the sorting action?"),
]

# ---------------------------------------------------------------------------
# Human prompt for Sort Dreaming (no image)
# ---------------------------------------------------------------------------
DREAM_PROMPT_SUFFIX = " What is the sorting action?"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_sort_dreaming(count: int = 500, seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples: list[dict] = []

    # 1. Add all explicit edge cases first
    for slug, condition, description in EDGE_CASES:
        part = SLUG_TO_CLASS[slug]
        action = apply_sorting_rules(part, condition)
        samples.append(_make_sample(description, slug, condition, action, len(samples)))

    # 2. Fill remainder with templated samples, balanced across classes & conditions
    remaining = count - len(samples)
    part_cycle = PART_CLASSES * (remaining // len(PART_CLASSES) + 1)
    random.shuffle(part_cycle)

    condition_weights = {
        "new":              0.15,
        "good":             0.15,
        "minor_wear":       0.15,
        "minor_corrosion":  0.20,
        "damaged":          0.20,
        "severely_damaged": 0.15,
    }
    cond_population = []
    for cond, w in condition_weights.items():
        cond_population.extend([cond] * int(w * remaining * 2))
    random.shuffle(cond_population)

    for i, part in enumerate(part_cycle[:remaining]):
        condition = cond_population[i % len(cond_population)]
        template = random.choice(TEMPLATES[condition])
        base_description = template(part.name)
        description = base_description + DREAM_PROMPT_SUFFIX

        action = apply_sorting_rules(part, condition)
        samples.append(_make_sample(description, part.slug, condition, action, len(samples)))

    random.shuffle(samples)
    # Re-index after shuffle
    for i, s in enumerate(samples):
        s["id"] = f"sort_dream_{i:04d}"

    return samples[:count]


def _make_sample(description: str, slug: str, condition: str, action, idx: int) -> dict:
    response = (
        f"Part: {slug}\n"
        f"Condition: {condition}\n"
        f"Action: bin={action.bin}, priority={action.priority}, "
        f"inspect={str(action.inspect).lower()}\n"
        f"Reason: {action.reason}"
    )
    return {
        "id": f"sort_dream_{idx:04d}",
        "image": None,   # text-only — no image
        "part_slug": slug,
        "condition": condition,
        "bin": action.bin,
        "priority": action.priority,
        "inspect": action.inspect,
        "conversations": [
            {"from": "human", "value": description},
            {"from": "gpt",   "value": response},
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def generate(
    count: int = typer.Option(500, help="Number of Sort Dreaming samples to generate"),
    seed: int = typer.Option(42),
    out_file: Path = typer.Option(DEFAULT_OUT, help="Output JSONL path"),
):
    """Generate the Sort Dreaming evaluation set."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_sort_dreaming(count=count, seed=seed)

    with jsonlines.open(out_file, mode="w") as w:
        w.write_all(samples)

    console.print(f"[green]Generated {len(samples)} Sort Dreaming samples → {out_file}[/green]")

    # Print a few examples
    console.print("\n[bold]Sample outputs:[/bold]")
    for s in samples[:3]:
        console.print(f"\n[cyan]{s['id']}[/cyan]")
        console.print(f"Q: {s['conversations'][0]['value'][:120]}...")
        console.print(f"A: {s['conversations'][1]['value']}")
        console.print("-" * 60)


def main():
    app()


if __name__ == "__main__":
    main()
