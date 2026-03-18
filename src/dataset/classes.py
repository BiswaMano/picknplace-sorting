"""
50 automotive part class definitions with bin mappings, category groupings,
and safety-critical flags for the VLA-AutoParts dataset.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Condition labels (ordered by severity)
# ---------------------------------------------------------------------------
CONDITIONS = [
    "new",
    "good",
    "minor_wear",
    "minor_corrosion",
    "damaged",
    "severely_damaged",
]

CONDITION_IDX = {c: i for i, c in enumerate(CONDITIONS)}

# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------
PRIORITIES = ["low", "normal", "high", "urgent"]
PRIORITY_IDX = {p: i for i, p in enumerate(PRIORITIES)}

# ---------------------------------------------------------------------------
# Bin layout (1-indexed)
#   1-3  : braking components
#   4-5  : suspension
#   6-7  : engine / drivetrain
#   8-9  : electrical / accessories
#   10   : reject / inspection hold
# ---------------------------------------------------------------------------
BIN_GROUPS = {
    "braking":    [1, 2, 3],
    "suspension": [4, 5],
    "engine":     [6, 7],
    "electrical": [8, 9],
    "reject":     [10],
}


# ---------------------------------------------------------------------------
# Part class definition
# ---------------------------------------------------------------------------
@dataclass
class PartClass:
    idx: int                    # 0-based index used as class label
    name: str                   # human-readable name
    slug: str                   # snake_case identifier used in filenames / JSON
    category: str               # braking | suspension | engine | electrical | misc
    default_bin: int            # bin for new/good condition
    safety_critical: bool       # elevates priority/inspect on any defect
    aliases: list[str] = field(default_factory=list)  # alternative names


# ---------------------------------------------------------------------------
# Master class list — 50 classes
# ---------------------------------------------------------------------------
PART_CLASSES: list[PartClass] = [
    # ── Braking (bins 1-3) ──────────────────────────────────────────────────
    PartClass(0,  "Brake Caliper",        "brake_caliper",       "braking",    1, True),
    PartClass(1,  "Brake Rotor",          "brake_rotor",         "braking",    1, True,  ["Brake Disc"]),
    PartClass(2,  "Brake Pad Set",        "brake_pad_set",       "braking",    2, True),
    PartClass(3,  "Brake Drum",           "brake_drum",          "braking",    2, True),
    PartClass(4,  "Brake Line",           "brake_line",          "braking",    3, True),

    # ── Suspension (bins 4-5) ───────────────────────────────────────────────
    PartClass(5,  "Wheel Bearing",        "wheel_bearing",       "suspension", 4, True),
    PartClass(6,  "CV Joint / Axle",      "cv_joint_axle",       "suspension", 4, True,  ["CV Axle"]),
    PartClass(7,  "Tie Rod End",          "tie_rod_end",         "suspension", 4, True),
    PartClass(8,  "Ball Joint",           "ball_joint",          "suspension", 4, True),
    PartClass(9,  "Control Arm",          "control_arm",         "suspension", 5, True),
    PartClass(10, "Shock Absorber",       "shock_absorber",      "suspension", 5, False),
    PartClass(11, "Strut Assembly",       "strut_assembly",      "suspension", 5, False),
    PartClass(12, "Coil Spring",          "coil_spring",         "suspension", 5, True),
    PartClass(13, "Sway Bar Link",        "sway_bar_link",       "suspension", 5, False),
    PartClass(14, "Wheel Hub",            "wheel_hub",           "suspension", 4, True),

    # ── Steering (suspension bin) ───────────────────────────────────────────
    PartClass(15, "Steering Rack",        "steering_rack",       "suspension", 5, True),
    PartClass(16, "Power Steering Pump",  "power_steering_pump", "suspension", 5, False),

    # ── Engine / Drivetrain (bins 6-7) ─────────────────────────────────────
    PartClass(17, "Alternator",           "alternator",          "engine",     6, False),
    PartClass(18, "Starter Motor",        "starter_motor",       "engine",     6, False),
    PartClass(19, "Ignition Coil",        "ignition_coil",       "engine",     6, False),
    PartClass(20, "Spark Plug",           "spark_plug",          "engine",     6, False),
    PartClass(21, "Fuel Injector",        "fuel_injector",       "engine",     6, False),
    PartClass(22, "Fuel Pump",            "fuel_pump",           "engine",     6, False),
    PartClass(23, "Oil Filter",           "oil_filter",          "engine",     7, False),
    PartClass(24, "Air Filter",           "air_filter",          "engine",     7, False),
    PartClass(25, "Cabin Filter",         "cabin_filter",        "engine",     7, False),
    PartClass(26, "Radiator",             "radiator",            "engine",     7, False),
    PartClass(27, "Water Pump",           "water_pump",          "engine",     7, False),
    PartClass(28, "Thermostat Housing",   "thermostat_housing",  "engine",     7, False),
    PartClass(29, "Serpentine Belt",      "serpentine_belt",     "engine",     7, False),
    PartClass(30, "Timing Belt",          "timing_belt",         "engine",     7, True),
    PartClass(31, "Exhaust Manifold",     "exhaust_manifold",    "engine",     7, False),
    PartClass(32, "Catalytic Converter",  "catalytic_converter", "engine",     7, False),
    PartClass(33, "Muffler",              "muffler",             "engine",     7, False),
    PartClass(34, "O2 Sensor",            "o2_sensor",           "engine",     6, False),
    PartClass(35, "MAF Sensor",           "maf_sensor",          "engine",     6, False),
    PartClass(36, "Throttle Body",        "throttle_body",       "engine",     6, False),
    PartClass(37, "EGR Valve",            "egr_valve",           "engine",     6, False),
    PartClass(38, "Turbocharger",         "turbocharger",        "engine",     7, False),
    PartClass(39, "Intake Manifold",      "intake_manifold",     "engine",     7, False),
    PartClass(40, "Valve Cover",          "valve_cover",         "engine",     7, False),
    PartClass(41, "Head Gasket",          "head_gasket",         "engine",     7, True),
    PartClass(42, "Flywheel",             "flywheel",            "engine",     7, True),
    PartClass(43, "Clutch Disc",          "clutch_disc",         "engine",     7, False),
    PartClass(44, "Transmission Filter",  "transmission_filter", "engine",     7, False),
    PartClass(45, "Differential Cover",   "differential_cover",  "engine",     7, False),
    PartClass(46, "Lug Nut Set",          "lug_nut_set",         "suspension", 4, True),

    # ── Electrical / Accessories (bins 8-9) ────────────────────────────────
    PartClass(47, "Heater Core",          "heater_core",         "electrical", 8, False),
    PartClass(48, "A/C Compressor",       "ac_compressor",       "electrical", 8, False),
    PartClass(49, "Windshield Wiper Motor","windshield_wiper_motor","electrical",9, False),
]

assert len(PART_CLASSES) == 50, f"Expected 50 classes, got {len(PART_CLASSES)}"

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
SLUG_TO_CLASS: dict[str, PartClass] = {p.slug: p for p in PART_CLASSES}
IDX_TO_CLASS:  dict[int, PartClass] = {p.idx:  p for p in PART_CLASSES}
NAME_TO_CLASS: dict[str, PartClass] = {p.name: p for p in PART_CLASSES}

CLASS_NAMES: list[str] = [p.name for p in PART_CLASSES]
CLASS_SLUGS: list[str] = [p.slug for p in PART_CLASSES]
NUM_CLASSES: int = len(PART_CLASSES)
NUM_BINS: int = 10
