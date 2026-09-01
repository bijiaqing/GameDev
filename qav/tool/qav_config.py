#!/usr/bin/env python3

"""Canonical backend-neutral definition of the GameDev QAV matrices"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

# QAV build and result trees are disposable; do not leave Python caches in source directories
sys.dont_write_bytecode = True


def model_executable(
    project_root: Path, model: str, backend: str, representation: str,
) -> Path:
    """Return one QAV executable from the generated-artifact tree"""

    candidates = (
        project_root/"mod"/model,
        project_root/"qav"/"comm"/representation/model,
        project_root/"qav"/backend/representation/model,
    )
    model_dirs = [path for path in candidates if (path/"flags.mk").is_file()]
    if len(model_dirs) != 1:
        raise RuntimeError(
            f"expected one flags directory for {model}, found {len(model_dirs)}"
        )
    return project_root/"qav"/"logs"/"build"/"bin"/backend/representation/model/"gamedev"


def model_analyzer(
    project_root: Path, representation: str, model: str, fallback: Callable,
) -> Callable:
    """Load an optional backend-neutral model validator or retain the backend default"""

    # keep coefficient-specialized references beside the common model so CUDA
    # and ROCm cannot silently select different expected solutions
    validator = project_root/"qav"/"comm"/representation/model/"validate_case.py"
    if not validator.is_file():
        return fallback

    module_name = f"qav_{representation}_{model}_validator"
    specification = importlib.util.spec_from_file_location(module_name, validator)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load model validator: {validator}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not isinstance(module, ModuleType) or not hasattr(module, "analyze"):
        raise RuntimeError(f"model validator does not define analyze: {validator}")
    return module.analyze


FLUID_GROUPS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "equilibrium": [("test_startup_3d", ())],
    "transport": [
        ("test_x_transport_2d", ("--shift", "3.25")),
        ("test_x_wedge_transport_2d", ()),
        ("test_y_transport_cyl", ("--cfl", "0.05")),
        ("test_y_transport_sph", ("--cfl", "0.05")),
        ("test_y_outflow_2d", ()),
        ("test_z_outflow_3d", ()),
        ("test_z_reflect_3d", ()),
        ("test_z_transport_3d", ("--cfl", "0.05")),
    ],
    "diffusion": [
        ("test_x_diffusion_2d", ()),
        ("test_x_wedge_diffusion_2d", ()),
        ("test_y_diffusion_cyl", ()),
        ("test_y_diffusion_sph", ()),
        ("test_z_diffusion_3d", ()),
        ("test_diffusion_poslimit", ()),
    ],
    "source": [("test_source_drag", ("--res", "8"))],
    "radiation": [
        ("test_optdepth", ("--power", "-1.0")),
        ("test_attenuation_2d", ("--power", "-1.0")),
    ],
    "coupled": [("test_ring_all_2d", ())],
}

SWARM_GROUPS: dict[str, list[str]] = {
    "transport": [
        "test_orbit_ecc_2d", "test_orbit_beta_2d", "test_orbit_inc_3d",
        "test_drag_path_1d", "test_prdrag_2d",
    ],
    "diffusion": ["test_diffusion_1d", "test_diffusion_2d", "test_diffusion_3d"],
    "initialization": ["test_initial_3d"],
    "collision": ["test_colphys_code", "test_colphys_cgs"],
    "knn": ["test_knn"],
}

SWARM_CHAIN_MODELS = [
    "test_colchain_2d",
    "test_colchain_frag_2d",
    "test_colchain_wedge_2d",
    "test_colchain_3d",
]

SWARM_FIXED_RESOLUTION = {
    "test_prdrag_2d", "test_colphys_code", "test_colphys_cgs", "test_knn",
}

SWARM_ENDPOINT_RESOLUTION = {"test_initial_3d"}

EXPECTED_FLUID_METRICS = 73
EXPECTED_SWARM_METRICS = 33
EXPECTED_PUBLICATION_FLUID_METRICS = EXPECTED_FLUID_METRICS
EXPECTED_PUBLICATION_SWARM_METRICS = EXPECTED_SWARM_METRICS

PUBLICATION_TIER = "publication"
QAV_TIERS = (PUBLICATION_TIER,)


def qav_output_path(qav_root: Path, requested: Path | None, default_name: str) -> Path:
    """Resolve an optional output name without allowing it to escape qav/logs"""

    logs_root = (qav_root/"logs").resolve()
    if requested is None:
        return logs_root/default_name

    direct = requested.expanduser().resolve()
    if direct.is_relative_to(logs_root):
        return direct
    if requested.is_absolute():
        raise ValueError(f"QAV output must remain below {logs_root}: {requested}")

    output = (logs_root/requested).resolve()
    if not output.is_relative_to(logs_root):
        raise ValueError(f"QAV output must remain below {logs_root}: {requested}")
    return output


def archive_fingerprint(root: Path) -> tuple[str, int]:
    """Hash the JSON and binary-field evidence below one relocatable archive directory"""

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".dat", ".json"}
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest(), len(files)


def source_fingerprint(project_root: Path) -> tuple[str, int]:
    """Hash every compiled or interpreted source that can affect a QAV result"""

    files = [project_root/"Makefile"]
    for root in (project_root/"inc", project_root/"src", project_root/"qav"):
        files.extend(
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {
                ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hip", ".hpp", ".inl", ".mk", ".py",
            }
            and "logs" not in path.parts
            and "__pycache__" not in path.parts
        )

    digest = hashlib.sha256()
    unique_files = sorted(set(files))
    for path in unique_files:
        relative = path.relative_to(project_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest(), len(unique_files)


def fluid_cases(group: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return one fluid group or the complete common matrix in canonical order"""

    if group == "all":
        return [case for name in FLUID_GROUPS for case in FLUID_GROUPS[name]]
    return list(FLUID_GROUPS[group])


def option_value(arguments: tuple[str, ...], option: str) -> str | None:
    """Return one option value from a flat command-line argument tuple"""

    try:
        index = arguments.index(option)
    except ValueError:
        return None
    return arguments[index + 1]


def fluid_case_tier(model: str, arguments: tuple[str, ...]) -> str:
    """Label every retained fluid case as publication evidence"""

    return PUBLICATION_TIER


def fluid_metric_tiers(
    cases: list[tuple[str, tuple[str, ...]]], resolution_count: int,
) -> dict[str, int]:
    """Count fluid metric records by minimum evidence tier"""

    counts = {PUBLICATION_TIER: 0}
    for model, arguments in cases:
        records = 1 if "--res" in arguments else resolution_count
        counts[fluid_case_tier(model, arguments)] += records
    return counts


def swarm_models(group: str) -> list[str]:
    """Return one swarm group or the complete common matrix in canonical order"""

    if group == "all":
        return [model for name in SWARM_GROUPS for model in SWARM_GROUPS[name]]
    return list(SWARM_GROUPS[group])


def swarm_model_tier(model: str) -> str:
    """Label every retained swarm model as publication evidence"""

    return PUBLICATION_TIER


def swarm_resolution_tiers(model: str, resolutions: list[int]) -> list[dict[str, int | str]]:
    """Label every retained analytical record as publication evidence"""

    return [
        {"resolution": resolution, "tier": PUBLICATION_TIER}
        for resolution in resolutions
    ]


def swarm_metric_tiers(models: list[str], resolutions: list[int]) -> dict[str, int]:
    """Count swarm analytical records by minimum evidence tier"""

    counts = {PUBLICATION_TIER: 0}
    for model in models:
        if model == "test_knn":
            continue
        model_resolutions = resolutions[:1] if model in SWARM_FIXED_RESOLUTION else (
            [resolutions[0], resolutions[-1]]
            if model in SWARM_ENDPOINT_RESOLUTION and len(resolutions) > 1
            else resolutions
        )
        for record in swarm_resolution_tiers(model, model_resolutions):
            counts[str(record["tier"])] += 1
    return counts
