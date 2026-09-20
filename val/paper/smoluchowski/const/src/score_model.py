#!/usr/bin/env python3

"""Reduce one completed constant-kernel model to compact JSON-ready scores."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import configparser
import json
from pathlib import Path

import numpy as np


LOG_MASS_EDGES = np.linspace(-0.5, 9.5, 201)
LOG_MASS_CDF_EDGES = np.linspace(-0.5, 9.5, 4097)


def _constant_mass_cdf(k: np.ndarray, g: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    result = np.zeros_like(k)
    valid = k >= 1.0
    log_tail = k[valid]*np.log1p(-g) + np.log1p(k[valid]*g)
    result[valid] = -np.expm1(log_tail)
    return np.clip(result, 0.0, 1.0)


def _analytic_cdf_below_log_mass(x: np.ndarray, g: float) -> np.ndarray:
    k_below = np.ceil(np.power(10.0, x)).astype(np.int64) - 1
    return _constant_mass_cdf(k_below, g)


def _jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5*(p + q)
    mask_p = p > 0.0
    mask_q = q > 0.0
    divergence = 0.5*np.sum(p[mask_p]*np.log(p[mask_p]/midpoint[mask_p]))
    divergence += 0.5*np.sum(q[mask_q]*np.log(q[mask_q]/midpoint[mask_q]))
    return float(np.sqrt(max(float(divergence), 0.0)/np.log(2.0)))


def score_model(output_dir: Path) -> dict[str, object]:
    config = configparser.ConfigParser()
    config.read(output_dir/"variables.txt")
    parameters = config["PARAMETERS"]
    seed = config.getint("CAMPAIGN", "SEED")
    dtype = np.dtype([(name, value) for name, value in config["SWARM_DTYPE"].items()])

    frame = int(parameters["SAVE_MAX"])
    time = float(parameters["DT_OUT"])*int(parameters["LOG_BASE"])**frame
    total_mass = float(parameters["TOTAL_DUST_MASS"])
    initial_mass = float(parameters["INIT_SMIN"])**3

    data = np.memmap(output_dir/f"particle_{frame:05d}.dat", dtype=dtype, mode="r")
    mass = np.asarray(data["par_size"])**3
    number = np.asarray(data["par_numr"])
    mass_weight = number*mass/total_mass
    mass_ratio = float(np.sum(mass_weight))
    log_mass = np.log10(mass/initial_mass)

    g = 1.0/(1.0 + 0.5*time)
    analytical_number = total_mass/initial_mass*g
    simulated_number = float(np.sum(number))

    coarse_index = np.searchsorted(LOG_MASS_EDGES, log_mass, side="right")
    coarse_probability = np.bincount(
        coarse_index, weights=mass_weight, minlength=len(LOG_MASS_EDGES) + 1,
    ).astype(np.float64)
    coarse_probability /= mass_ratio

    analytic_cdf_edges = _analytic_cdf_below_log_mass(LOG_MASS_EDGES, g)
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
    analytical_cdf = _analytic_cdf_below_log_mass(LOG_MASS_CDF_EDGES, g)
    cdf_difference = np.abs(simulated_cdf - analytical_cdf)
    w1_log_mass = np.sum(
        0.5*(cdf_difference[:-1] + cdf_difference[1:])*np.diff(LOG_MASS_CDF_EDGES)
    )

    simulated_m2_over_m1 = float(np.sum(mass_weight*mass)/mass_ratio)
    analytical_m2_over_m1 = initial_mass*(1.0 + time)
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
        "seed": seed,
        "parameters": {
            "N_P": int(parameters["N_P"]),
            "N_K": int(parameters["N_K"]),
            "COL_BATH_EPS": float(parameters["COL_BATH_EPS"]),
        },
        "runtime_seeds": {
            "position": config.getint("CAMPAIGN", "POSITION_SEED"),
            "collision": config.getint("CAMPAIGN", "COLLISION_SEED"),
        },
        "score": score,
        "histograms": histograms,
    }


def scoring_metadata() -> dict[str, object]:
    return {
        "snapshot": "final particle frame only",
        "distribution": "physical-mass probability",
        "grain_mass": "par_size**3",
        "analytic_mass_probability": "p_k = k*g**2*(1-g)**(k-1), g = 1/(1+t/2)",
        "coarse_log10_mass_edges": LOG_MASS_EDGES.tolist(),
        "fine_log10_mass_edges": LOG_MASS_CDF_EDGES.tolist(),
        "histogram_end_bins": "first is underflow; last is overflow",
        "primary_shape_metric": "total-variation distance on coarse bins",
        "w1_unit": "dex in log10(m/m0)",
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = score_model(args.output_dir)
    result["reference"] = scoring_metadata()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.json)
