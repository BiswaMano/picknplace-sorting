"""
Web image downloader for VLA-AutoParts dataset.

Downloads images for each of the 50 automotive part classes from
DuckDuckGo Images and/or Bing Images — no API key required.

Search queries are carefully crafted to return high-quality product/part
images rather than installed-in-vehicle shots (which are harder to
isolate to a single part class).

Usage:
  python -m src.dataset.image_downloader               # all 50 classes
  python -m src.dataset.image_downloader --classes brake_caliper brake_rotor
  python -m src.dataset.image_downloader --per-class 60 --engine bing
  python -m src.dataset.image_downloader --check       # count existing images
"""

import shutil
import time
import hashlib
import random
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from src.dataset.classes import PART_CLASSES, SLUG_TO_CLASS, CLASS_SLUGS

app = typer.Typer(help="Download web images for VLA-AutoParts dataset")
console = Console()

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Search queries per part class
# Queries are tuned to return isolated product images, not in-vehicle shots.
# Multiple queries per class increase variety.
# ---------------------------------------------------------------------------

SEARCH_QUERIES: dict[str, list[str]] = {
    "brake_caliper":        ["brake caliper auto part isolated", "brake caliper product photo white background", "automotive brake caliper used"],
    "brake_rotor":          ["brake rotor disc auto part", "brake disc product photo isolated", "automotive brake rotor"],
    "brake_pad_set":        ["brake pad set auto part", "brake pads product photo", "ceramic brake pads isolated"],
    "brake_drum":           ["brake drum auto part isolated", "rear brake drum product photo", "automotive brake drum"],
    "brake_line":           ["brake line automotive part", "brake hose product photo", "stainless steel brake line"],
    "wheel_bearing":        ["wheel bearing hub auto part", "wheel bearing assembly product", "automotive wheel bearing isolated"],
    "cv_joint_axle":        ["cv joint axle auto part", "cv axle shaft product photo", "constant velocity joint automotive"],
    "tie_rod_end":          ["tie rod end auto part isolated", "tie rod end product photo", "steering tie rod end"],
    "ball_joint":           ["ball joint auto part isolated", "automotive ball joint product", "front lower ball joint"],
    "control_arm":          ["control arm auto part isolated", "lower control arm product photo", "suspension control arm"],
    "shock_absorber":       ["shock absorber auto part isolated", "shock absorber product photo", "automotive shock strut"],
    "strut_assembly":       ["strut assembly auto part", "complete strut assembly product", "front strut automotive"],
    "coil_spring":          ["coil spring auto part isolated", "suspension coil spring product", "automotive coil spring"],
    "sway_bar_link":        ["sway bar link auto part", "stabilizer bar link product photo", "end link sway bar"],
    "wheel_hub":            ["wheel hub assembly auto part", "wheel hub bearing product photo", "automotive wheel hub"],
    "steering_rack":        ["steering rack auto part isolated", "rack and pinion product photo", "power steering rack"],
    "power_steering_pump":  ["power steering pump auto part", "power steering pump product photo", "hydraulic power steering pump"],
    "alternator":           ["alternator auto part isolated", "remanufactured alternator product", "automotive alternator 12v"],
    "starter_motor":        ["starter motor auto part isolated", "starter motor product photo", "automotive starter"],
    "ignition_coil":        ["ignition coil auto part", "coil pack product photo", "automotive ignition coil"],
    "spark_plug":           ["spark plug auto part isolated", "spark plug product photo", "NGK iridium spark plug"],
    "fuel_injector":        ["fuel injector auto part isolated", "fuel injector product photo", "direct injection injector"],
    "fuel_pump":            ["fuel pump auto part isolated", "electric fuel pump product", "automotive fuel pump module"],
    "oil_filter":           ["oil filter auto part isolated", "oil filter product photo", "spin on oil filter"],
    "air_filter":           ["air filter auto part isolated", "engine air filter product", "automotive air cleaner filter"],
    "cabin_filter":         ["cabin air filter auto part", "cabin filter product photo", "pollen filter interior"],
    "radiator":             ["car radiator auto part isolated", "automotive radiator product photo", "aluminum radiator replacement"],
    "water_pump":           ["water pump auto part isolated", "automotive water pump product", "engine coolant pump"],
    "thermostat_housing":   ["thermostat housing auto part", "coolant thermostat product photo", "engine thermostat housing"],
    "serpentine_belt":      ["serpentine belt auto part", "drive belt product photo", "poly-v belt automotive"],
    "timing_belt":          ["timing belt auto part isolated", "timing belt kit product", "cam belt automotive"],
    "exhaust_manifold":     ["exhaust manifold auto part", "header exhaust manifold product", "cast iron exhaust manifold"],
    "catalytic_converter":  ["catalytic converter auto part", "cat converter product photo", "OEM catalytic converter"],
    "muffler":              ["muffler auto part isolated", "exhaust muffler product photo", "automotive muffler silencer"],
    "o2_sensor":            ["oxygen sensor auto part isolated", "o2 sensor product photo", "lambda sensor automotive"],
    "maf_sensor":           ["MAF sensor auto part isolated", "mass airflow sensor product", "air flow meter automotive"],
    "throttle_body":        ["throttle body auto part isolated", "electronic throttle body product", "throttle valve assembly"],
    "egr_valve":            ["EGR valve auto part isolated", "exhaust gas recirculation valve product", "EGR valve automotive"],
    "turbocharger":         ["turbocharger auto part isolated", "turbo charger product photo", "turbocharger replacement"],
    "intake_manifold":      ["intake manifold auto part isolated", "intake manifold product photo", "aluminum intake manifold"],
    "valve_cover":          ["valve cover auto part isolated", "cam cover product photo", "rocker cover automotive"],
    "head_gasket":          ["head gasket auto part isolated", "cylinder head gasket product", "MLS head gasket automotive"],
    "flywheel":             ["flywheel auto part isolated", "dual mass flywheel product", "automotive flywheel"],
    "clutch_disc":          ["clutch disc auto part isolated", "clutch plate product photo", "organic clutch disc"],
    "transmission_filter":  ["transmission filter auto part", "ATF filter product photo", "automatic transmission filter"],
    "differential_cover":   ["differential cover auto part", "rear diff cover product photo", "axle cover automotive"],
    "lug_nut_set":          ["lug nut set auto part isolated", "wheel lug nuts product photo", "locking lug nut set"],
    "heater_core":          ["heater core auto part isolated", "heater matrix product photo", "automotive heater core"],
    "ac_compressor":        ["AC compressor auto part isolated", "air conditioning compressor product", "automotive AC compressor"],
    "windshield_wiper_motor":["wiper motor auto part isolated", "windshield wiper motor product", "wiper drive motor automotive"],
}


# ---------------------------------------------------------------------------
# Download engine wrappers
# ---------------------------------------------------------------------------

def _download_ddg(query: str, save_dir: Path, limit: int, existing: int) -> int:
    """Download via DuckDuckGo Images (uses ddgs package)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return 0

    downloaded = 0
    retries = 3
    for attempt in range(retries):
        try:
            with DDGS() as ddg:
                results = list(ddg.images(
                    query,
                    max_results=limit * 3,   # fetch extra to account for failures
                ))

            for r in results:
                if downloaded >= limit:
                    break
                url = r.get("image", "")
                if not url:
                    continue
                downloaded += _fetch_and_save(url, save_dir, existing + downloaded)
                time.sleep(0.15)   # polite delay
            break  # success — exit retry loop

        except Exception as e:
            err_str = str(e)
            if "Ratelimit" in err_str or "403" in err_str:
                wait = (attempt + 1) * 3
                time.sleep(wait)   # back off on rate limit
            else:
                console.print(f"    [yellow]DDG error: {e}[/yellow]")
                break

    return downloaded


def _download_bing(query: str, save_dir: Path, limit: int, existing: int) -> int:
    """Download via icrawler Bing engine."""
    try:
        from icrawler.builtin import BingImageCrawler
        import logging
        logging.disable(logging.CRITICAL)   # suppress icrawler logs
    except ImportError:
        return 0

    try:
        crawler = BingImageCrawler(
            storage={"root_dir": str(save_dir)},
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=limit, min_size=(200, 200))
        # Count new files
        new_count = len(list(save_dir.glob("*.jpg"))) + len(list(save_dir.glob("*.png"))) - existing
        return max(0, new_count)
    except Exception as e:
        console.print(f"    [yellow]Bing error: {e}[/yellow]")
        return 0


def _fetch_and_save(url: str, save_dir: Path, idx: int) -> int:
    """Fetch URL, validate as image, save. Returns 1 on success, 0 on failure."""
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = resp.read()

        # Validate as image using PIL
        from io import BytesIO
        img = Image.open(BytesIO(data)).convert("RGB")
        w, h = img.size
        if min(w, h) < 150:
            return 0   # too small

        # Save with a hash-based filename to avoid duplicates
        h_str = hashlib.md5(data).hexdigest()[:10]
        dst = save_dir / f"web_{idx:04d}_{h_str}.jpg"
        if dst.exists():
            return 0
        img.save(dst, "JPEG", quality=90)
        return 1

    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-class downloader
# ---------------------------------------------------------------------------

def download_class(
    slug: str,
    per_class: int = 60,
    engine: str = "ddg",
    raw_dir: Path = RAW_DIR,
) -> int:
    """Download images for a single part class. Returns count of new images."""
    part = SLUG_TO_CLASS.get(slug)
    if part is None:
        console.print(f"[red]Unknown slug: {slug}[/red]")
        return 0

    save_dir = raw_dir / slug
    save_dir.mkdir(parents=True, exist_ok=True)

    # Count already existing images
    existing = len(list(save_dir.glob("*.jpg"))) + len(list(save_dir.glob("*.png")))
    needed = max(0, per_class - existing)
    if needed == 0:
        console.print(f"  {slug}: already has {existing} images — skipping")
        return 0

    queries = SEARCH_QUERIES.get(slug, [f"{part.name} automotive part isolated"])
    total_new = 0

    for query in queries:
        if total_new >= needed:
            break
        batch_limit = max(1, (needed - total_new) // max(1, len(queries) - queries.index(query)) + 5)

        if engine == "ddg":
            n = _download_ddg(query, save_dir, batch_limit, existing + total_new)
            # Fall back to Bing if DDG returned nothing
            if n == 0:
                n = _download_bing(query, save_dir, batch_limit, existing + total_new)
        else:
            n = _download_bing(query, save_dir, batch_limit, existing + total_new)
            if n == 0:
                n = _download_ddg(query, save_dir, batch_limit, existing + total_new)

        total_new += n
        time.sleep(random.uniform(0.5, 1.5))   # rate limiting

    return total_new


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def download(
    classes: list[str] = typer.Argument(
        None, help="Part slugs to download (default: all 50)"
    ),
    per_class: int = typer.Option(60, help="Target images per class"),
    engine: str = typer.Option("ddg", help="Search engine: ddg | bing"),
    raw_dir: Path = typer.Option(RAW_DIR, help="Raw data output directory"),
):
    """Download web images for all (or specified) part classes."""
    slugs = classes if classes else CLASS_SLUGS

    console.print(f"[bold]Downloading images for {len(slugs)} classes[/bold]")
    console.print(f"  Engine: {engine} | Target: {per_class}/class | Dir: {raw_dir}")
    console.print()

    summary: dict[str, int] = {}

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(slugs))

        for slug in slugs:
            progress.update(task, description=f"[cyan]{slug}")
            n = download_class(slug, per_class=per_class, engine=engine, raw_dir=raw_dir)
            summary[slug] = n
            progress.advance(task)

    # Print summary table
    table = Table(title="Download Summary")
    table.add_column("Part Class")
    table.add_column("New Downloads", justify="right")
    table.add_column("Total in dir", justify="right")

    for slug in slugs:
        part_dir = raw_dir / slug
        total = len(list(part_dir.glob("*.jpg"))) + len(list(part_dir.glob("*.png"))) if part_dir.exists() else 0
        new = summary.get(slug, 0)
        color = "green" if total >= 50 else "yellow"
        table.add_row(slug, str(new), f"[{color}]{total}[/{color}]")

    console.print(table)


@app.command()
def check(raw_dir: Path = typer.Option(RAW_DIR)):
    """Count existing downloaded images per class."""
    table = Table(title="Current Image Counts")
    table.add_column("Part Class")
    table.add_column("Count", justify="right")
    table.add_column("Status")

    for part in PART_CLASSES:
        part_dir = raw_dir / part.slug
        count = 0
        if part_dir.exists():
            count = len(list(part_dir.glob("*.jpg"))) + len(list(part_dir.glob("*.png")))
        status = "[green]OK[/green]" if count >= 50 else f"[yellow]need {50 - count} more[/yellow]"
        table.add_row(part.name, str(count), status)

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
