#!/usr/bin/env python3

"""Canonical backend-neutral definition of the GameDev QAV matrices"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable


def model_executable(
    project_root: Path, model: str, backend: str, representation: str,
) -> Path:
    """Return the executable beside the unique flags file selected for one model"""

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
    return model_dirs[0]/"gamedev"


def model_analyzer(
    project_root: Path, representation: str, model: str, fallback: Callable,
) -> Callable:
    """Load an optional backend-neutral model validator or retain the backend default"""

    # load the shared backend validator for historical cases, but keep coefficient-specialized references beside the common
    # model so CUDA and ROCm cannot silently select different expected solutions
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
    "transport": [
        *(('test_x_transport_2d', ('--shift', str(shift)))
          for shift in (3.0, 3.25, 3.5, 3.75)),
        ("test_y_transport_cyl", ("--cfl", "0.05")),
        ("test_y_transport_cyl", ("--cfl", "0.5")),
        ("test_y_transport_sph", ("--cfl", "0.05")),
        ("test_y_transport_sph", ("--cfl", "0.5")),
        ("test_y_outflow_2d", ()),
        ("test_y_outflow_3d", ()),
        ("test_z_transport_3d", ("--cfl", "0.05")),
        ("test_z_transport_3d", ("--cfl", "0.5")),
    ],
    "diffusion": [
        ("test_x_diffusion_2d", ()),
        ("test_y_diffusion_cyl", ()),
        ("test_y_diffusion_sph", ()),
        ("test_z_diffusion_3d", ()),
        ("test_diffusion_poslimit", ()),
    ],
    "source": [("test_source_drag", ("--res", "8"))],
    "radiation": [
        *(('test_optdepth', ('--power', str(power))) for power in (0.0, -1.0, 1.0)),
        *(('test_attenuation_2d', ('--power', str(power))) for power in (-1.0, 1.0)),
    ],
    "ring": [
        ("test_ring_transport_2d", ()),
        ("test_ring_diffusion_2d", ()),
        ("test_ring_radiation_2d", ()),
        ("test_ring_all_2d", ()),
    ],
}

SWARM_GROUPS: dict[str, list[str]] = {
    "grid": ["test_grid_1d", "test_grid_2d", "test_grid_3d"],
    "transport": [
        "test_orbit_1d", "test_drag_1d", "test_viscflow_1d",
        "test_orbit_2d", "test_drag_2d", "test_orbit_ecc_2d", "test_orbit_beta_2d",
        "test_orbit_inc_3d", "test_drag_path_1d",
        "test_absorb_path_1d",
    ],
    "diffusion": [
        "test_diffusion_1d", "test_diffusion_2d", "test_diffusion_3d", "test_settle_diffuse_3d",
    ],
    "initialization": ["test_initial_3d"],
    "radiation": [
        "test_radiation_1d", "test_prdrag_1d", "test_radiation_2d", "test_prdrag_2d",
    ],
    "boundary": [
        "test_boundary_1d", "test_boundary_2d", "test_boundary_3d", "test_boundary_half",
    ],
    "collision": [
        "test_collision_1d", "test_import_1d", "test_collision_2d", "test_collision_3d",
    ],
    "knn": ["test_knn"],
}

SWARM_RADIAL_MODELS = [
    "test_grid_1d",
    "test_orbit_1d",
    "test_drag_1d",
    "test_drag_path_1d",
    "test_absorb_path_1d",
    "test_viscflow_1d",
    "test_diffusion_1d",
    "test_radiation_1d",
    "test_prdrag_1d",
    "test_boundary_1d",
    "test_collision_1d",
    "test_import_1d",
    "test_knn",
]

SWARM_FIXED_RESOLUTION = {
    "test_drag_1d",
    "test_viscflow_1d",
    "test_radiation_1d",
    "test_prdrag_1d",
    "test_collision_1d",
    "test_import_1d",
    "test_drag_2d",
    "test_radiation_2d",
    "test_prdrag_2d",
    "test_collision_2d",
    "test_collision_3d",
    "test_boundary_1d",
    "test_boundary_2d",
    "test_boundary_3d",
    "test_boundary_half",
    "test_knn",
}

EXPECTED_FLUID_METRICS = 105
EXPECTED_SWARM_METRICS = 75
EXPECTED_PUBLICATION_FLUID_METRICS = 77
EXPECTED_PUBLICATION_SWARM_METRICS = 57

PUBLICATION_TIER = "publication"
RELEASE_TIER = "release"
QUALIFICATION_TIER = "qualification"
QAV_TIERS = (PUBLICATION_TIER, RELEASE_TIER, QUALIFICATION_TIER)

SWARM_RELEASE_MODELS = {
    "test_boundary_1d",
    "test_boundary_2d",
    "test_boundary_3d",
    "test_boundary_half",
}

SWARM_QUALIFICATION_MODELS = {"test_failure_knn", "test_restart_2d"}

SWARM_SINGLE_PUBLICATION_RESOLUTION = {
    "test_grid_1d",
    "test_grid_2d",
    "test_grid_3d",
    "test_orbit_1d",
}


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
    """Classify one fluid parameter variant by its minimum evidence tier"""

    if model == "test_x_transport_2d" and option_value(arguments, "--shift") != "3.25":
        return RELEASE_TIER
    if model in {"test_y_transport_cyl", "test_y_transport_sph", "test_z_transport_3d"} \
            and option_value(arguments, "--cfl") != "0.05":
        return RELEASE_TIER
    if model == "test_optdepth" and option_value(arguments, "--power") == "0.0":
        return RELEASE_TIER
    return PUBLICATION_TIER


def fluid_metric_tiers(
    cases: list[tuple[str, tuple[str, ...]]], resolution_count: int,
) -> dict[str, int]:
    """Count fluid metric records by minimum evidence tier"""

    counts = {PUBLICATION_TIER: 0, RELEASE_TIER: 0}
    for model, arguments in cases:
        records = 1 if "--res" in arguments else resolution_count
        counts[fluid_case_tier(model, arguments)] += records
    return counts


def swarm_models(group: str) -> list[str]:
    """Return one swarm group or the complete common matrix in canonical order"""

    if group == "all":
        return [model for name in SWARM_GROUPS for model in SWARM_GROUPS[name]]
    if group == "radial":
        return list(SWARM_RADIAL_MODELS)
    return list(SWARM_GROUPS[group])


def swarm_model_tier(model: str) -> str:
    """Classify one swarm model by its minimum evidence tier"""

    if model in SWARM_QUALIFICATION_MODELS:
        return QUALIFICATION_TIER
    return RELEASE_TIER if model in SWARM_RELEASE_MODELS else PUBLICATION_TIER


def swarm_resolution_tiers(model: str, resolutions: list[int]) -> list[dict[str, int | str]]:
    """Label analytical records while retaining release-only repetition tests"""

    publication = set(resolutions)
    if model in SWARM_QUALIFICATION_MODELS:
        return [
            {"resolution": resolution, "tier": QUALIFICATION_TIER}
            for resolution in resolutions
        ]
    if model in SWARM_RELEASE_MODELS:
        publication.clear()
    elif model in SWARM_SINGLE_PUBLICATION_RESOLUTION and resolutions:
        publication = {resolutions[-1]}
    elif model == "test_initial_3d" and resolutions:
        publication = {resolutions[0], resolutions[-1]}
    return [
        {
            "resolution": resolution,
            "tier": PUBLICATION_TIER if resolution in publication else RELEASE_TIER,
        }
        for resolution in resolutions
    ]


def swarm_metric_tiers(models: list[str], resolutions: list[int]) -> dict[str, int]:
    """Count swarm analytical records by minimum evidence tier"""

    counts = {PUBLICATION_TIER: 0, RELEASE_TIER: 0}
    for model in models:
        if model == "test_knn":
            continue
        model_resolutions = resolutions[:1] if model in SWARM_FIXED_RESOLUTION else resolutions
        for record in swarm_resolution_tiers(model, model_resolutions):
            counts[str(record["tier"])] += 1
    return counts
