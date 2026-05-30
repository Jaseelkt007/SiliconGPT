#!/usr/bin/env python3
"""
generate_sequences.py — Semiconductor process sequence generator and validator.

Encodes the process grammar from generation_rules.md and produces valid
synthetic sequences for MOSFET, IGBT, and IC product families.

Usage
-----
Generate 500 MOSFET variants:
    python generate_sequences.py --family mosfet --count 500 --output MOSFET_variants.csv --seed 42

Generate all families (auto-count from combinatorics):
    python generate_sequences.py --family igbt --output IGBT_variants.csv --seed 42

Validate an existing sequence file against all 10 process-logic rules:
    python generate_sequences.py --validate mysequences.csv --family mosfet

Print combinatoric estimate without generating:
    python generate_sequences.py --family ic --estimate-only
"""

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Vocabulary sets used by the validator
# ---------------------------------------------------------------------------

# Steps that require a clean surface before them (RULE_DEP_NO_CLEAN).
DEPOSITION_STEPS = frozenset({
    "THERMAL OXIDATION",
    "GATE OXIDE GROWTH",
    "DEPOSIT PAD OXIDE",
    "EPITAXIAL DEPOSITION",
    "DEPOSIT POLYSILICON",
    "DEPOSIT SPACER DIELECTRIC",
    "DEPOSIT FIELD OXIDE",
    "DEPOSIT GATE OXIDE OR DIELECTRIC",
    "DEPOSIT INTERLAYER DIELECTRIC",
    "DEPOSIT INTERLEVEL DIELECTRIC",
    "DEPOSIT BARRIER METAL",
    "DEPOSIT METAL SEED",
    "DEPOSIT METAL 1",
    "DEPOSIT TOP METAL",
    "DEPOSIT BACKSIDE METAL",
    "DEPOSIT TUNGSTEN SEED",
    "DEPOSIT PASSIVATION",
    "DEPOSIT PASSIVATION LAYER",
    "DEPOSIT BACKSIDE PROTECTION",
})

# Steps that create a clean / well-defined surface for the next deposition.
# Note: THERMAL OXIDATION is in both sets — it requires clean and creates clean.
CLEAN_STEPS = frozenset({
    # Wet / chemical cleans
    "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
    "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
    "HF DIP", "OXIDE STRIP", "SURFACE PREP FOR DEPOSITION",
    "FRONTSIDE CLEAN", "BACKSIDE CLEAN", "FRONTSIDE CLEAN FINAL",
    "BACKSIDE CLEAN FINAL", "WAFER CLEAN PRE-GRIND",
    "DRY WAFER", "DRY WAFER BACKSIDE",
    # Post-etch cleans
    "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH",
    "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH",
    "CLEAN AFTER WINDOW ETCH", "CLEAN AFTER FIELD ETCH",
    "CLEAN PAD OPENING", "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
    # Thermal steps that create a clean/passivated surface
    "THERMAL OXIDATION",    # grows clean oxide; subsequent poly sees clean SiO2
    "GATE OXIDE PREP",      # surface conditioning before gate oxide
    "RAPID THERMAL ANNEAL", # high-temp anneal in inert/oxidising atmosphere
    "EPITAXY ANNEAL",       # post-epitaxy anneal; clean atmosphere
    "ANNEAL OXIDE",         # oxide anneal
})

# Patterned etch steps that require a lithography develop step before them.
# Note: ANISOTROPIC ETCH SPACER is deliberately excluded — it is a blanket etch.
ETCH_STEPS = frozenset({
    "OXIDE ETCH", "OXIDE ETCH DRY",
    "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
    "ETCH SILICON OR OXIDE WINDOW",
    "FIELD OXIDE ETCH",
    "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA",
    "METAL ETCH", "METAL ETCH DRY",
    "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH",
})

METAL_ETCH_STEPS = frozenset({"METAL ETCH", "METAL ETCH DRY"})

IMPLANT_STEPS = frozenset({
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION",
    "IMPLANT LDD", "IMPLANT P BODY", "IMPLANT N BUFFER",
    "IMPLANT CHANNEL STOP", "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
})

# A patterned opening (oxide etch or litho develop) must precede an implant.
IMPLANT_OPENER_STEPS = frozenset({
    "OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
    "DEVELOP PHOTORESIST",
})

CMP_STEPS = frozenset({
    "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC",
    "CMP METAL", "CMP VIA FILL",
})

# Steps that produce a filled surface that CMP can act on.
FILL_STEPS = frozenset({
    "FILL VIA METAL", "FILL VIA TUNGSTEN",
}) | DEPOSITION_STEPS

PAD_WINDOW_STEPS = frozenset({
    "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW",
    "PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO",
})

ELECTRICAL_TEST_STEPS = frozenset({
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
})

BACKSIDE_METAL_STEPS = frozenset({"DEPOSIT BACKSIDE METAL"})


# ---------------------------------------------------------------------------
# Validator — checks all 10 process-logic rules
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    rule: str
    description: str
    step_index: int
    step_name: str

    def __str__(self) -> str:
        return f"[{self.rule}] step {self.step_index} ({self.step_name!r}): {self.description}"


def validate_sequence(steps: list[str]) -> list[Violation]:
    """
    Check a sequence against all 10 process-logic rules.
    Returns a list of Violation objects; empty list means the sequence is valid.
    """
    violations: list[Violation] = []

    def window(i: int, size: int) -> list[str]:
        return steps[max(0, i - size):i]

    def any_in_window(i: int, size: int, targets: frozenset) -> bool:
        return any(s in targets for s in window(i, size))

    # ------------------------------------------------------------------ #
    # RULE_DEP_NO_CLEAN                                                    #
    # Every deposition step must have a clean step within the prior 12.   #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in DEPOSITION_STEPS:
            if not any_in_window(i, 12, CLEAN_STEPS):
                violations.append(Violation(
                    rule="RULE_DEP_NO_CLEAN",
                    description=(
                        f"Deposition step '{step}' has no clean step in the prior 12 steps. "
                        f"A clean surface is required before any deposition."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_METAL_ETCH_NO_LITHO                                             #
    # Metal etch requires EXPOSE LITHO + DEVELOP within prior 15 steps.   #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in METAL_ETCH_STEPS:
            w = window(i, 15)
            has_expose = any(s.startswith("EXPOSE LITHO LEVEL") for s in w)
            has_develop = ("DEVELOP PHOTORESIST" in w) or ("DEVELOP PAD WINDOW" in w)
            if not (has_expose and has_develop):
                violations.append(Violation(
                    rule="RULE_METAL_ETCH_NO_LITHO",
                    description=(
                        f"Metal etch '{step}' is missing EXPOSE LITHO LEVEL or DEVELOP PHOTORESIST "
                        f"in the prior 15 steps. Metal cannot be etched without a photoresist mask."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_ETCH_NO_MASK                                                    #
    # Every patterned etch requires DEVELOP PHOTORESIST within prior 12.  #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in ETCH_STEPS:
            w = window(i, 12)
            has_develop = ("DEVELOP PHOTORESIST" in w) or ("DEVELOP PAD WINDOW" in w)
            if not has_develop:
                violations.append(Violation(
                    rule="RULE_ETCH_NO_MASK",
                    description=(
                        f"Etch step '{step}' has no DEVELOP PHOTORESIST in the prior 12 steps. "
                        f"A photoresist mask must be patterned before etching."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_LITHO_LEVEL_SKIP                                                #
    # Litho mask levels must appear in ascending sequential order.        #
    # ------------------------------------------------------------------ #
    align_steps = [
        (i, int(step.split("ALIGN MASK LEVEL ")[1]))
        for i, step in enumerate(steps)
        if step.startswith("ALIGN MASK LEVEL ")
        and step.split("ALIGN MASK LEVEL ")[1].isdigit()
    ]
    for idx in range(1, len(align_steps)):
        prev_i, prev_lvl = align_steps[idx - 1]
        curr_i, curr_lvl = align_steps[idx]
        if curr_lvl > prev_lvl + 1:
            violations.append(Violation(
                rule="RULE_LITHO_LEVEL_SKIP",
                description=(
                    f"Litho level jumps from {prev_lvl} (step {prev_i}) to {curr_lvl} "
                    f"(step {curr_i}), skipping level {prev_lvl + 1}."
                ),
                step_index=curr_i,
                step_name=steps[curr_i],
            ))
        if curr_lvl < prev_lvl:
            violations.append(Violation(
                rule="RULE_LITHO_LEVEL_SKIP",
                description=(
                    f"Litho level decreases from {prev_lvl} (step {prev_i}) to {curr_lvl} "
                    f"(step {curr_i}). Levels must be non-decreasing."
                ),
                step_index=curr_i,
                step_name=steps[curr_i],
            ))

    # ------------------------------------------------------------------ #
    # RULE_IMPLANT_NO_MASK                                                 #
    # Every implant must have an oxide etch or litho develop within 15.   #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in IMPLANT_STEPS:
            if not any_in_window(i, 15, IMPLANT_OPENER_STEPS):
                violations.append(Violation(
                    rule="RULE_IMPLANT_NO_MASK",
                    description=(
                        f"Implant step '{step}' has no oxide etch or DEVELOP PHOTORESIST "
                        f"in the prior 15 steps. An open implant window is required."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_CMP_NO_DEP                                                      #
    # Every CMP step must have a deposition or fill step within prior 6.  #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in CMP_STEPS:
            if not any_in_window(i, 6, FILL_STEPS):
                violations.append(Violation(
                    rule="RULE_CMP_NO_DEP",
                    description=(
                        f"CMP step '{step}' has no deposition or fill step in the prior 6 steps. "
                        f"There must be material to planarize."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_PAD_OPEN_BEFORE_DEP                                             #
    # Pad window steps must appear after DEPOSIT PASSIVATION and CURE.    #
    # ------------------------------------------------------------------ #
    passivation_dep_idx: Optional[int] = None
    cure_passivation_idx: Optional[int] = None
    for i, step in enumerate(steps):
        if step in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"):
            passivation_dep_idx = i
        if step == "CURE PASSIVATION":
            cure_passivation_idx = i
        if step in PAD_WINDOW_STEPS:
            if passivation_dep_idx is None or i < passivation_dep_idx:
                violations.append(Violation(
                    rule="RULE_PAD_OPEN_BEFORE_DEP",
                    description=(
                        f"Pad window step '{step}' at index {i} appears before "
                        f"DEPOSIT PASSIVATION (index {passivation_dep_idx}). "
                        f"You cannot open a window in passivation that has not been deposited."
                    ),
                    step_index=i,
                    step_name=step,
                ))
            elif cure_passivation_idx is None or i < cure_passivation_idx:
                violations.append(Violation(
                    rule="RULE_PAD_OPEN_BEFORE_DEP",
                    description=(
                        f"Pad window step '{step}' at index {i} appears before "
                        f"CURE PASSIVATION (index {cure_passivation_idx}). "
                        f"Passivation must be cured before the pad window is opened."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_TEST_BEFORE_PASSIVATION                                         #
    # Electrical tests must appear after CURE PASSIVATION.               #
    # ------------------------------------------------------------------ #
    cure_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None
    )
    for i, step in enumerate(steps):
        if step in ELECTRICAL_TEST_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append(Violation(
                    rule="RULE_TEST_BEFORE_PASSIVATION",
                    description=(
                        f"Electrical test '{step}' at index {i} appears before "
                        f"CURE PASSIVATION (index {cure_idx}). "
                        f"Devices must be passivated before electrical characterization."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    # ------------------------------------------------------------------ #
    # RULE_SHIP_BEFORE_TEST                                                #
    # SHIP LOT must appear after WAFER SORT TEST.                        #
    # ------------------------------------------------------------------ #
    ship_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "SHIP LOT"), None
    )
    sort_idx: Optional[int] = next(
        (i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None
    )
    if ship_idx is not None and (sort_idx is None or ship_idx < sort_idx):
        violations.append(Violation(
            rule="RULE_SHIP_BEFORE_TEST",
            description=(
                f"SHIP LOT at index {ship_idx} appears before "
                f"WAFER SORT TEST (index {sort_idx}). "
                f"Lots must pass sort testing before they can be shipped."
            ),
            step_index=ship_idx,
            step_name="SHIP LOT",
        ))

    # ------------------------------------------------------------------ #
    # RULE_BACKSIDE_BEFORE_PASSIVATION                                     #
    # DEPOSIT BACKSIDE METAL must appear after CURE PASSIVATION.         #
    # ------------------------------------------------------------------ #
    for i, step in enumerate(steps):
        if step in BACKSIDE_METAL_STEPS:
            if cure_idx is None or i < cure_idx:
                violations.append(Violation(
                    rule="RULE_BACKSIDE_BEFORE_PASSIVATION",
                    description=(
                        f"'{step}' at index {i} appears before "
                        f"CURE PASSIVATION (index {cure_idx}). "
                        f"The frontside must be passivated before backside metallization."
                    ),
                    step_index=i,
                    step_name=step,
                ))

    return violations


# ---------------------------------------------------------------------------
# Grammar helpers
# ---------------------------------------------------------------------------

def _opt(rng: random.Random, step: str, prob: float = 0.75) -> list[str]:
    """Return [step] with probability `prob`, else []."""
    return [step] if rng.random() < prob else []


def _pre_anneal(rng: random.Random) -> list[str]:
    return ["PRE ANNEAL CHECK"] if rng.random() > 0.4 else []


def _meas(rng: random.Random, step: str, prob: float = 0.75) -> list[str]:
    return [step] if rng.random() < prob else []


def _litho(rng: random.Random, level: int, inspection: Optional[str] = None) -> list[str]:
    """Generate one lithography block for mask level `level`."""
    steps = [
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        f"ALIGN MASK LEVEL {level}",
        f"EXPOSE LITHO LEVEL {level}",
    ]
    steps += _opt(rng, "POST EXPOSE BAKE", 0.3)
    steps.append("DEVELOP PHOTORESIST")
    steps.append(inspection or f"INSPECT PATTERN LEVEL {level}")
    steps += _opt(rng, "HARD BAKE", 0.3)
    return steps


# ---------------------------------------------------------------------------
# Block generators — shared
# ---------------------------------------------------------------------------

def _gen_prefix(rng: random.Random) -> list[str]:
    return [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
    ]


def _gen_initial_measurements(rng: random.Random, family: str) -> list[str]:
    steps = []
    thickness = {
        "mosfet": ["MEASURE THICKNESS"],
        "igbt":   ["MEASURE INITIAL THICKNESS"],
        "ic":     ["MEASURE INITIAL GEOMETRY", "MEASURE INITIAL THICKNESS"],
    }[family]
    if rng.random() > 0.15:
        steps.append(rng.choice(thickness))
    surface = {
        "mosfet": ["MEASURE SURFACE PARTICLES"],
        "igbt":   ["MEASURE SURFACE PARTICLES"],
        "ic":     ["MEASURE SURFACE DEFECTS", "MEASURE SURFACE PARTICLES"],
    }[family]
    if rng.random() > 0.15:
        steps.append(rng.choice(surface))
    return steps


def _gen_pre_process_clean(rng: random.Random, family: str) -> list[str]:
    steps: list[str] = []
    steps.append("WAFER CLEAN PRE PROCESS" if family == "ic" else "PRE CLEAN WAFER")
    # IGBT always has separate backside/frontside clean; others optionally
    if family == "igbt" or rng.random() > 0.5:
        steps.append("BACKSIDE CLEAN")
    if family == "igbt" or rng.random() > 0.6:
        steps.append("FRONTSIDE CLEAN")
    steps.append(rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]))
    steps.append(rng.choice(["RCA CLEAN 2", "WET CLEAN RCA2"]))
    steps.append("HF DIP")
    steps += _opt(rng, "DRY WAFER", 0.6)
    return steps


def _gen_ild_block(rng: random.Random) -> list[str]:
    return [
        rng.choice(["DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"]),
        rng.choice(["DENSIFY DIELECTRIC", "DENSIFY OXIDE"]),
        rng.choice(["MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"]),
        rng.choice(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"]),
        rng.choice(["MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"]),
    ]


def _gen_via_fill(rng: random.Random, family: str) -> list[str]:
    if family == "ic":
        s = [
            "DEPOSIT BARRIER METAL",
            "DEPOSIT TUNGSTEN SEED",
            "FILL VIA TUNGSTEN",
            rng.choice(["CMP VIA FILL", "CMP METAL"]),
        ]
    else:
        s = [
            "DEPOSIT BARRIER METAL",
            "DEPOSIT METAL SEED",
            "FILL VIA METAL",
            rng.choice(["CMP METAL", "CMP VIA FILL"]),
        ]
    s += _meas(rng, rng.choice(["MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"]))
    return s


def _gen_via_block(rng: random.Random, level: int, family: str) -> list[str]:
    """Via litho cycle + etch + fill."""
    s = _litho(rng, level, rng.choice(["VIA INSPECTION", "VIA OPENING INSPECTION"]))
    s += [
        rng.choice(["VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"]),
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN AFTER VIA ETCH",
    ]
    s += _meas(rng, "MEASURE VIA CD")
    s += _gen_via_fill(rng, family)
    return s


def _gen_metal_block(rng: random.Random, level: int, family: str) -> list[str]:
    """Metal deposition + litho + metal etch."""
    metal_dep = rng.choice(["DEPOSIT METAL 1", "DEPOSIT TOP METAL"])
    metal_ann = rng.choice(["ANNEAL METAL 1", "ANNEAL METAL"])
    etch_step = "METAL ETCH DRY" if family in ("igbt", "ic") else "METAL ETCH"
    s = [metal_dep, metal_ann]
    s += _meas(rng, "MEASURE METAL THICKNESS", 0.45)
    s += _litho(rng, level, "METAL PATTERN INSPECTION")
    s += [
        etch_step,
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER METAL ETCH",
    ]
    s += _meas(rng, "MEASURE LINE WIDTH")
    return s


def _gen_passivation_block(rng: random.Random) -> list[str]:
    return [
        rng.choice(["DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"]),
        "CURE PASSIVATION",
        rng.choice(["MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"]),
        rng.choice(["OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"]),
        rng.choice(["PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"]),
        rng.choice(["DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"]),
        rng.choice(["PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN PAD OPENING",
        "MEASURE PAD OPENING",
    ]


def _gen_backside_block(rng: random.Random, family: str) -> list[str]:
    if family == "ic":
        # IC thinned early; this block is just final backside protect + anneal
        s = []
        s += _opt(rng, "BACKSIDE THINNING CHECK", 0.6)
        s.append(rng.choice(["BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL"]))
        s += ["DEPOSIT BACKSIDE PROTECTION", "BACKSIDE ANNEAL"]
        s += _opt(rng, "FRONTSIDE CLEAN FINAL", 0.5)
        return s
    # MOSFET / IGBT: grind happens here
    s: list[str] = []
    if family == "mosfet":
        s.append("BACKSIDE CLEAN")
    s += [
        "BACKSIDE GRIND",
        rng.choice(["MEASURE THICKNESS", "MEASURE WAFER THICKNESS"]),
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "BACKSIDE DRY",
        "BACKSIDE METALLIZATION PREP",
        "DEPOSIT BACKSIDE METAL",
        "BACKSIDE ANNEAL",
        "MEASURE BACKSIDE CONTACT",
    ]
    return s


def _gen_final_inspection(rng: random.Random, family: str) -> list[str]:
    s = ["FINAL CLEAN"]
    s += _meas(rng, "FINAL THICKNESS MEASURE", 0.8)
    s += _meas(rng, "FINAL GEOMETRY CHECK", 0.8)
    if family == "ic":
        s += _meas(rng, "FINAL OXIDE CHECK", 0.55)
    s += _meas(rng, "FINAL CD INSPECTION", 0.5)
    s += _meas(rng, "FINAL PARTICLE INSPECTION", 0.8)
    if family == "ic":
        s += _opt(rng, "FINAL ELECTRICAL TEST PREP", 0.5)
    return s


def _gen_test_suite(rng: random.Random, family: str) -> list[str]:
    param = rng.choice(["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"])
    family_test = {
        "mosfet": "THRESHOLD VOLTAGE TEST",
        "igbt":   "BREAKDOWN VOLTAGE TEST",
        "ic":     rng.choice(["THRESHOLD VOLTAGE TEST", "PARAMETRIC TEST"]),
    }[family]
    s = [param, "LEAKAGE TEST"]
    if family_test != param:
        s.append(family_test)
    s.append("SWITCHING TEST")
    # IGBT reference places YIELD ANALYSIS before WAFER SORT TEST in some variants
    if family == "igbt" and rng.random() > 0.5:
        s += ["YIELD ANALYSIS", "WAFER SORT TEST"]
    else:
        s += ["WAFER SORT TEST", "YIELD ANALYSIS"]
    return s


def _gen_suffix(rng: random.Random, family: str) -> list[str]:
    s = [rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"])]
    if family == "ic":
        s += _opt(rng, "PACKAGE PREPARATION", 0.5)
    s.append("SHIP LOT")
    return s


# ---------------------------------------------------------------------------
# Family-specific block generators
# ---------------------------------------------------------------------------

def _gen_family_prep_mosfet(rng: random.Random) -> list[str]:
    return [
        "SUBSTRATE CHECK",
        "EPITAXY PREP",
        "EPITAXIAL DEPOSITION",
        "MEASURE EPITAXY THICKNESS",
        "MEASURE RESISTIVITY",
        "EPITAXY ANNEAL",
        "WAFER SURFACE CLEAN",  # clean surface before gate oxidation block
    ]


def _gen_family_prep_igbt(rng: random.Random) -> list[str]:
    s = ["EPITAXIAL WAFER CHECK", "MEASURE EPITAXY THICKNESS", "MEASURE RESISTIVITY"]
    s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.5)
    s.append("EPITAXIAL LAYER PREP")
    return s


def _gen_family_prep_ic(rng: random.Random) -> list[str]:
    return [
        "WAFER CLEAN PRE-GRIND",
        "GRINDING WAFER BACKSIDE",
        rng.choice(["MEASURE GEOMETRY", "MEASURE INITIAL GEOMETRY"]),
        "ETCH WET BACKSIDE",
        "RINSE WET WAFER_EDGE",
        "DRY WAFER BACKSIDE",
        "BACKSIDE CLEAN",
        "MEASURE BACKSIDE ROUGHNESS",
    ]


def _gen_first_oxidation(rng: random.Random, family: str) -> list[str]:
    s = ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    if family == "ic":
        # IC re-cleans after pad-oxide prep before depositing pad oxide
        s.append(rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]))
        s.append(rng.choice(["RCA CLEAN 2", "WET CLEAN RCA2"]))
        s += [
            "HF DIP",
            "OXIDE STRIP",
            "SURFACE PREP FOR DEPOSITION",
            "DEPOSIT PAD OXIDE",
            "ANNEAL OXIDE",
            rng.choice(["MEASURE FILM THICKNESS", "MEASURE OXIDE THICKNESS"]),
        ]
    return s


# ---------------------------------------------------------------------------
# Process cycles (the core repeated litho–etch–implant logic)
# ---------------------------------------------------------------------------

def _gen_cycles_mosfet(rng: random.Random) -> list[str]:
    s: list[str] = []

    # Cycle 1 — well implant (mask level 1)
    s += _litho(rng, 1, "PATTERN INSPECTION LEVEL 1")
    s += ["OXIDE ETCH", rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]), "CLEAN AFTER ETCH"]
    s += _meas(rng, "MEASURE OPENING CD")
    s += ["IMPLANT WELL"] + _pre_anneal(rng)
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE JUNCTION DEPTH")

    # Gate oxide block (THERMAL OXIDATION creates clean surface for poly)
    s += ["THERMAL OXIDATION", "GATE OXIDE PREP", "GATE OXIDE GROWTH", "MEASURE GATE OXIDE THICKNESS"]

    # Cycle 2 — poly gate + source/drain implant (mask level 2)
    s += ["DEPOSIT POLYSILICON", "POLYSILICON ANNEAL"]
    s += _meas(rng, "MEASURE POLY THICKNESS")
    s += _litho(rng, 2, "POLY PATTERN INSPECTION")
    s += ["POLYSILICON ETCH", rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]), "CLEAN AFTER POLY ETCH"]
    s += _meas(rng, "MEASURE GATE CD")
    s += ["IMPLANT SOURCE DRAIN"] + _pre_anneal(rng)
    s += ["LIGHT ANNEAL"]
    s += _meas(rng, "MEASURE SHEET RESISTANCE")

    # Spacer + LDD sub-block (blanket etch — no litho needed for spacer etch)
    s += ["DEPOSIT SPACER DIELECTRIC", "ANISOTROPIC ETCH SPACER"]
    s += _meas(rng, "MEASURE SPACER WIDTH")
    s += ["IMPLANT LDD"] + _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE JUNCTION PROFILE")

    return s


def _gen_cycles_igbt(rng: random.Random) -> list[str]:
    s: list[str] = []

    # Cycle 1 — P body implant (mask level 1)
    s += _litho(rng, 1, "INSPECT PATTERN LEVEL 1")
    s += ["OXIDE ETCH DRY", rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]), "CLEAN AFTER OXIDE ETCH"]
    s += _meas(rng, "MEASURE OPENING CD")
    s += ["IMPLANT P BODY"] + _pre_anneal(rng)
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE JUNCTION DEPTH")

    # Cycle 2 — N buffer window (mask level 2)
    s += ["THERMAL OXIDATION"]
    s += _litho(rng, 2, "P BODY WINDOW INSPECTION")
    s += [
        "ETCH SILICON OR OXIDE WINDOW",
        rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]),
        "CLEAN AFTER WINDOW ETCH",
    ]
    s += _meas(rng, "MEASURE WINDOW CD")
    s += ["IMPLANT N BUFFER"] + _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE SHEET RESISTANCE")
    s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.4)

    # Field oxide block
    s += ["DEPOSIT FIELD OXIDE", "DENSIFY OXIDE"]
    s += _meas(rng, "MEASURE FILM THICKNESS")

    # Cycle 3 — field oxide etch + source/drain/cathode implants (mask level 3)
    s += _litho(rng, 3, "FIELD PATTERN INSPECTION")
    s += ["FIELD OXIDE ETCH", rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]), "CLEAN AFTER FIELD ETCH"]
    s += _meas(rng, "MEASURE SURFACE UNIFORMITY")
    s += ["IMPLANT SOURCE REGION", "IMPLANT DRAIN / CATHODE REGION"] + _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE SHEET RESISTANCE")

    # Gate dielectric + poly (RAPID THERMAL ANNEAL above provides clean surface)
    s += ["DEPOSIT GATE OXIDE OR DIELECTRIC", "ANNEAL DIELECTRIC"]
    s += _meas(rng, "MEASURE OXIDE QUALITY")
    s += ["DEPOSIT POLYSILICON", "POLYSILICON ANNEAL"]
    s += _meas(rng, "MEASURE POLY THICKNESS")

    # Cycle 4 — poly gate patterning (mask level 4)
    s += _litho(rng, 4, "POLY PATTERN INSPECTION")
    s += ["POLYSILICON ETCH DRY", rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]), "CLEAN AFTER POLY ETCH"]
    s += _meas(rng, "MEASURE GATE CD")
    s += ["IMPLANT CHANNEL STOP"] + _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE DEVICE PARAMETER")

    return s


def _gen_cycles_ic(rng: random.Random) -> list[str]:
    s: list[str] = []

    # Cycle 1 — pad oxide / STI patterning (mask level 1)
    litho1 = _litho(rng, 1, "INSPECT PATTERN LEVEL 1")
    if "HARD BAKE" not in litho1:
        litho1.append("HARD BAKE")   # IC cycle 1 always has hard bake
    s += litho1
    s += ["OXIDE ETCH DRY", rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]), "CLEAN AFTER ETCH"]
    s += _meas(rng, "MEASURE CD LEVEL 1")

    # Poly deposition (CLEAN AFTER ETCH above provides clean surface)
    s += ["DEPOSIT POLYSILICON", "ANNEAL POLYSILICON"]

    # Cycle 2 — poly gate patterning + N-type implant (mask level 2)
    s += _litho(rng, 2, "PATTERN INSPECTION LEVEL 2")
    s += ["POLYSILICON ETCH DRY", rng.choice(["STRIP RESIST LEVEL 2", "STRIP RESIST"]), "CLEAN AFTER POLY ETCH"]
    s += _meas(rng, "MEASURE CD LEVEL 2")
    s += ["IMPLANT N-TYPE"] + _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE SHEET RESISTANCE")

    return s


_CYCLE_GEN = {
    "mosfet": _gen_cycles_mosfet,
    "igbt":   _gen_cycles_igbt,
    "ic":     _gen_cycles_ic,
}
_FAMILY_PREP = {
    "mosfet": _gen_family_prep_mosfet,
    "igbt":   _gen_family_prep_igbt,
    "ic":     _gen_family_prep_ic,
}
_VIA_LEVEL  = {"mosfet": 3, "igbt": 5, "ic": 3}
_METAL_LEVEL = {"mosfet": 4, "igbt": 6, "ic": 4}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sequence(family: str, rng: random.Random) -> list[str]:
    """
    Generate one valid process sequence for the given product family.

    Parameters
    ----------
    family : str
        One of "mosfet", "igbt", "ic" (case-insensitive).
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    list[str]
        Ordered list of step names.
    """
    family = family.lower()
    if family not in _CYCLE_GEN:
        raise ValueError(f"Unknown family '{family}'. Choose from: mosfet, igbt, ic")

    steps: list[str] = []
    steps += _gen_prefix(rng)
    steps += _gen_initial_measurements(rng, family)
    steps += _gen_pre_process_clean(rng, family)
    steps += _FAMILY_PREP[family](rng)
    steps += _gen_first_oxidation(rng, family)
    steps += _CYCLE_GEN[family](rng)
    steps += _gen_ild_block(rng)
    steps += _gen_via_block(rng, _VIA_LEVEL[family], family)
    steps += _gen_metal_block(rng, _METAL_LEVEL[family], family)
    steps += _gen_passivation_block(rng)
    steps += _gen_backside_block(rng, family)
    steps += _gen_final_inspection(rng, family)
    steps += _gen_test_suite(rng, family)
    steps += _gen_suffix(rng, family)
    return steps


def generate_dataset(
    family: str,
    count: int,
    seed: int = 42,
    validate: bool = True,
) -> list[list[str]]:
    """
    Generate `count` unique sequences for `family`.

    Parameters
    ----------
    family : str
        Product family (mosfet / igbt / ic).
    count : int
        Number of unique sequences to produce.
    seed : int
        Base random seed.
    validate : bool
        If True, each sequence is checked against all 10 rules and rejected
        if invalid. Invalid sequences should not occur from the grammar; a
        warning is printed if one is found so the grammar can be fixed.

    Returns
    -------
    list[list[str]]
        List of step sequences.
    """
    rng = random.Random(seed)
    sequences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = count * 20

    while len(sequences) < count and attempts < max_attempts:
        attempts += 1
        seq = generate_sequence(family, rng)
        key = tuple(seq)
        if key in seen:
            continue
        if validate:
            viol = validate_sequence(seq)
            if viol:
                print(
                    f"[WARN] Grammar produced an invalid sequence "
                    f"({viol[0].rule}: {viol[0].description}). "
                    "Skipping — please report this as a grammar bug.",
                    file=sys.stderr,
                )
                continue
        seen.add(key)
        sequences.append(seq)

    if len(sequences) < count:
        print(
            f"[WARN] Only generated {len(sequences)}/{count} unique sequences "
            f"after {attempts} attempts. "
            "Try reducing --count or increasing combinatoric variation.",
            file=sys.stderr,
        )
    return sequences


def estimate_combinatorics(family: str) -> int:
    """
    Estimate the number of structurally distinct sequences for a family.

    Counts independent binary variation axes (optional steps) and
    discrete synonym choices. The true count is a lower bound because
    interactions between axes are not modelled.
    """
    family = family.lower()

    # Litho block: POST EXPOSE BAKE (2) × HARD BAKE (2) per cycle
    cycles = {"mosfet": 4, "igbt": 6, "ic": 4}[family]
    litho_variants = (2 * 2) ** cycles

    # PRE ANNEAL CHECK occurrences
    anneal_points = {"mosfet": 4, "igbt": 5, "ic": 2}[family]
    anneal_variants = 2 ** anneal_points

    # Optional measurement steps
    meas_points = {"mosfet": 9, "igbt": 10, "ic": 7}[family]
    meas_variants = 2 ** meas_points

    # Pre-process clean options: backside (2) × frontside (2) × dry (2)
    # × RCA1 synonym (2) × RCA2 synonym (2)
    clean_variants = 2 ** 5

    # Family-specific optionals
    family_opts = {"mosfet": 0, "igbt": 2, "ic": 1}[family]
    family_variants = 2 ** family_opts

    # Synonym choices in key positions (via etch, strip, suffix, etc.)
    synonym_variants = 3 * 2 * 2 * 2 * 2

    # Inspection label choices, metal/anneal synonyms
    misc_variants = 2 * 2 * 2 * 2

    total = (litho_variants * anneal_variants * meas_variants *
             clean_variants * family_variants * synonym_variants * misc_variants)
    return total


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_csv(path: Path, sequences: list[list[str]]) -> None:
    """Write sequences to CSV with SEQUENCE_ID and STEP columns."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SEQUENCE_ID", "STEP"])
        for i, seq in enumerate(sequences, start=1):
            seq_id = f"seq_{i:04d}"
            for step in seq:
                writer.writerow([seq_id, step])
    print(f"  Wrote {len(sequences)} sequences "
          f"({sum(len(s) for s in sequences):,} total step rows) -> {path}")


def read_csv_sequences(path: Path) -> dict[str, list[str]]:
    """
    Read a CSV file produced by write_csv (SEQUENCE_ID + STEP columns).
    Also handles:
    - Legacy single-column files (STEP only = one sequence).
    - UTF-8 BOM prefix on the first column header.
    - Column names wrapped in double-quotes as literal characters.
    Returns {sequence_id: [step, ...]}.
    """

    def _normalise(name: str) -> str:
        """Strip BOM, surrounding quotes, and whitespace from a header name."""
        return name.lstrip("\ufeff").strip().strip('"').strip()

    sequences: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_fields = reader.fieldnames or []
        # Build a mapping from normalised name → original header key
        norm_map = {_normalise(h): h for h in raw_fields}
        has_seq_id = "SEQUENCE_ID" in norm_map
        has_step   = "STEP" in norm_map

        if not has_step:
            raise ValueError(
                f"Cannot parse {path}: expected a 'STEP' column "
                f"(found headers: {raw_fields})."
            )

        step_key = norm_map["STEP"]
        seq_key  = norm_map.get("SEQUENCE_ID")

        for row in reader:
            step_val = row[step_key].strip().strip('"')
            if not step_val:
                continue
            if has_seq_id and seq_key:
                sid = row[seq_key].strip()
            else:
                sid = "seq_0001"
            sequences.setdefault(sid, []).append(step_val)

    return sequences


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _cmd_generate(args: argparse.Namespace) -> None:
    family = args.family.lower()
    count = args.count
    seed = args.seed
    output = Path(args.output or f"{family.upper()}_variants.csv")

    est = estimate_combinatorics(family)
    print(f"Combinatoric estimate for {family.upper()}: ~{est:,} distinct sequences")

    if count > est:
        print(
            f"[WARN] Requested count {count} exceeds estimate {est:,}. "
            "Duplicate-free generation may not be possible; using best-effort.",
            file=sys.stderr,
        )

    print(f"Generating {count} sequences (seed={seed}) ...")
    sequences = generate_dataset(family, count, seed=seed, validate=True)
    write_csv(output, sequences)


def _cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.validate)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    sequences = read_csv_sequences(path)
    total, invalid_count = 0, 0
    for sid, steps in sequences.items():
        total += 1
        violations = validate_sequence(steps)
        if violations:
            invalid_count += 1
            print(f"\n  Sequence {sid!r} ({len(steps)} steps) — {len(violations)} violation(s):")
            for v in violations:
                print(f"    {v}")
    print(f"\nValidated {total} sequence(s): "
          f"{total - invalid_count} valid, {invalid_count} invalid.")


def _cmd_estimate(args: argparse.Namespace) -> None:
    family = args.family.lower()
    est = estimate_combinatorics(family)
    print(f"Estimated structurally distinct sequences for {family.upper()}: ~{est:,}")
    print("(Actual unique duplicate-free count may be lower due to RNG sampling overlap.)")


# ---------------------------------------------------------------------------
# Argument parser and entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_sequences.py",
        description="Semiconductor process sequence generator and validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--family",
        choices=["mosfet", "igbt", "ic"],
        metavar="FAMILY",
        help="Product family: mosfet | igbt | ic",
    )
    p.add_argument(
        "--count", type=int, default=500,
        help="Number of sequences to generate (default: 500).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output CSV path (defaults to <FAMILY>_variants.csv).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    p.add_argument(
        "--validate", metavar="CSV_FILE",
        help="Validate all sequences in an existing CSV file against the 10 process-logic rules.",
    )
    p.add_argument(
        "--estimate-only", action="store_true",
        help="Print combinatoric estimate for --family and exit without generating.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.validate:
        _cmd_validate(args)
        return

    if not args.family:
        parser.error("--family is required unless using --validate.")

    if args.estimate_only:
        _cmd_estimate(args)
        return

    _cmd_generate(args)


if __name__ == "__main__":
    main()
