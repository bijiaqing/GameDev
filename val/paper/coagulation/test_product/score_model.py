#!/usr/bin/env python3

"""Reduce one completed product-kernel model to compact JSON-ready scores."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import configparser
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


LOG_MASS_EDGES = np.linspace(-0.5, 9.5, 201)
LOG_MASS_CDF_EDGES = np.linspace(-0.5, 9.5, 4097)
REFERENCE_K_MAX = 100_000


@lru_cache(maxsize=4)
def _borel_cdf(time: float) -> np.ndarray:
    probability = np.empty(REFERENCE_K_MAX, dtype=np.float64)
    probability[0] = math.exp(-time)
    common = time*math.exp(-time)
    for k in range(1, REFERENCE_K_MAX):
        probability[k] = probability[k - 1]*common*math.exp((k - 1)*math.log1p(1.0/k))
    return np.cumsum(probability)


def _analytic_cdf_below_log_mass(x: np.ndarray, time: float) -> np.ndarray:
    k_below = np.ceil(np.power(10.0, x)).astype(np.int64) - 1
    cdf = _borel_cdf(time)
    result = np.zeros_like(x, dtype=np.float64)
    valid = k_below >= 1
    result[valid] = cdf[np.minimum(k_below[valid], len(cdf)) - 1]
    return np.clip(result, 0.0, 1.0)


def _jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5*(p + q)
    mask_p = p > 0.0
    mask_q = q > 0.0
    divergence = 0.5*np.sum(p[mask_p]*np.log(p[mask_p]/midpoint[mask_p]))
    divergence += 0.5*np.sum(q[mask_q]*np.log(q[mask_q]/midpoint[mask_q]))
    return float(np.sqrt(max(float(divergence), 0.0)/np.log(2.0)))


def _controller_summary(output_dir: Path) -> dict[str, object]:
    sum_fields = (
        "operator_count", "bath_count", "continuation_launches",
        "activity_overshoots", "distribution_overshoots", "persistent_overshoots",
    )
    min_fields = ("minimum_duration", "minimum_limit_scale")
    max_fields = (
        "maximum_duration", "maximum_f", "maximum_e", "maximum_touched",
        "maximum_events", "maximum_g", "maximum_g_upper", "maximum_d_bath",
    )

    frames: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("collision_chain_*.json")):
        with path.open() as file:
            raw = json.load(file)
        frame = int(path.stem.rsplit("_", 1)[1])
        frames.append({"frame": frame, **{key: raw[key] for key in (*sum_fields, *min_fields, *max_fields)}})

    total: dict[str, object] = {key: sum(int(frame[key]) for frame in frames) for key in sum_fields}
    total.update({key: min(float(frame[key]) for frame in frames) for key in min_fields})
    total.update({key: max(float(frame[key]) for frame in frames) for key in max_fields})
    return {"frames": frames, "total": total}


def score_model(output_dir: Path) -> dict[str, object]:
    config = configparser.ConfigParser()
    config.read(output_dir/"variables.txt")
    parameters = config["PARAMETERS"]
    dtype = np.dtype([(name, value) for name, value in config["SWARM_DTYPE"].items()])

    frame = int(parameters["SAVE_MAX"])
    time = float(parameters["DT_OUT"])*frame
    total_mass = float(parameters["TOTAL_DUST_MASS"])
    initial_mass = float(parameters["INIT_SMIN"])**3

    data = np.memmap(output_dir/f"particle_{frame:05d}.dat", dtype=dtype, mode="r")
    mass = np.asarray(data["par_size"])**3
    number = np.asarray(data["par_numr"])
    mass_weight = number*mass/total_mass
    mass_ratio = float(np.sum(mass_weight))
    log_mass = np.log10(mass/initial_mass)

    analytical_number = total_mass/initial_mass*(1.0 - 0.5*time)
    simulated_number = float(np.sum(number))

    coarse_index = np.searchsorted(LOG_MASS_EDGES, log_mass, side="right")
    coarse_probability = np.bincount(
        coarse_index, weights=mass_weight, minlength=len(LOG_MASS_EDGES) + 1,
    ).astype(np.float64)
    coarse_probability /= mass_ratio

    analytic_cdf_edges = _analytic_cdf_below_log_mass(LOG_MASS_EDGES, time)
    analytic_probability = np.diff(np.concatenate(([0.0], analytic_cdf_edges, [1.0])))
    analytic_probability = np.clip(analytic_probability, 0.0, None)
    analytic_probability /= np.sum(analytic_probability)

    tv_distance = 0.5*np.sum(np.abs(coarse_probability - analytic_probability))
    js_distance = _jensen_shannon_distance(coarse_probability, analytic_probability)

    fine_index = np.searchsorted(LOG_MASS_CDF_EDGES, log_mass, side="right")
    fine_probability = np.bincount(
        fine_index, weights=mass_weight, minlength=len(LOG_MASS_CDF_EDGES) + 1,
    ).astype(np.float64)
    fine_probability /= mass_ratio
    simulated_cdf = np.cumsum(fine_probability)[:-1]
    analytical_cdf = _analytic_cdf_below_log_mass(LOG_MASS_CDF_EDGES, time)
    cdf_difference = np.abs(simulated_cdf - analytical_cdf)
    w1_log_mass = np.sum(
        0.5*(cdf_difference[:-1] + cdf_difference[1:])*np.diff(LOG_MASS_CDF_EDGES)
    )

    simulated_m2_over_m1 = float(np.sum(mass_weight*mass)/mass_ratio)
    analytical_m2_over_m1 = initial_mass/(1.0 - time)
    number_ratio = simulated_number/analytical_number
    moment_ratio = simulated_m2_over_m1/analytical_m2_over_m1

    score = {
        "frame": frame,
        "time": time,
        "tv_distance": float(tv_distance),
        "js_distance": js_distance,
        "w1_log10_mass_dex": float(w1_log_mass),
        "cdf_sup_distance": float(np.max(cdf_difference)),
        "mass_ratio": mass_ratio,
        "mass_relative_error": abs(mass_ratio - 1.0),
        "number_ratio": number_ratio,
        "number_relative_error": abs(number_ratio - 1.0),
        "m2_over_m1_ratio": moment_ratio,
        "m2_over_m1_relative_error": abs(moment_ratio - 1.0),
        "minimum_mass": float(np.min(mass)),
        "maximum_mass": float(np.max(mass)),
        "maximum_integer_mass_error": float(np.max(np.abs(mass - np.rint(mass)))),
        "simulated_underflow_mass": float(coarse_probability[0]),
        "simulated_overflow_mass": float(coarse_probability[-1]),
        "analytical_underflow_mass": float(analytic_probability[0]),
        "analytical_overflow_mass": float(analytic_probability[-1]),
    }
    histograms = {
        "coarse_mass_probability": coarse_probability.tolist(),
        "fine_mass_probability": fine_probability.tolist(),
    }
    del data

    return {
        "parameters": {
            "N_P": int(parameters["N_P"]),
            "N_K": int(parameters["N_K"]),
            "COL_BATH_EPS": float(parameters["COL_BATH_EPS"]),
        },
        "runtime_seeds": {
            "position": int(parameters["POSITION_SEED"]),
            "collision": int(parameters["COLLISION_SEED"]),
            "partner_permutation": int(parameters["COLLISION_PARTNER_SEED"]),
        },
        "score": score,
        "histograms": histograms,
        "controller": _controller_summary(output_dir),
    }


def scoring_metadata() -> dict[str, object]:
    return {
        "snapshot": "final particle frame only",
        "distribution": "physical-mass probability",
        "grain_mass": "par_size**3",
        "analytic_mass_probability": "p_k = exp(-k*t)*(k*t)**(k-1)/k!",
        "analytic_regime": "monodisperse product kernel before gelation, t < 1",
        "borel_reference_max_integer_mass": REFERENCE_K_MAX,
        "coarse_log10_mass_edges": LOG_MASS_EDGES.tolist(),
        "fine_log10_mass_edges": LOG_MASS_CDF_EDGES.tolist(),
        "histogram_end_bins": "first is underflow; last is overflow",
        "primary_shape_metric": "total-variation distance on coarse bins",
        "w1_unit": "dex in log10(m/m0)",
    }
