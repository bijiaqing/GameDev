#!/usr/bin/env python3

"""Check the merged CUDA and ROCm source tree without requiring either compiler"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import val_config
from val_config import (
    EXPECTED_FLUID_METRICS,
    EXPECTED_PUBLICATION_FLUID_METRICS,
    EXPECTED_PUBLICATION_SWARM_METRICS,
    EXPECTED_SWARM_METRICS,
    FLUID_GROUPS,
    SWARM_ENDPOINT_RESOLUTION,
    SWARM_FIXED_RESOLUTION,
    SWARM_GROUPS,
    fluid_metric_tiers,
    swarm_metric_tiers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PRODUCTION_UNITS = {
    "fluid": {
        "advection_xbl", "advection_xth", "advection_ybl", "advection_yth",
        "advection_zbl", "advection_zth", "cfl_rate_calc", "diffusion_xbl",
        "diffusion_xth", "diffusion_ybl", "diffusion_yth", "diffusion_zbl",
        "diffusion_zth", "fluid_runtime", "inf_cell_flag", "init_rho_calc",
        "init_vel_calc", "momentum_getv", "momentum_setv", "optdepth_calc",
        "optdepth_csum", "source_update",
    },
    "swarm": {
        "col_event_run", "col_rate_calc", "col_site_init", "col_snap_save",
        "diffusion_pos", "dustdens_calc", "dustdens_depo", "dustdens_init",
        "dyn_rate_calc", "gas_lerp_calc", "optdepth_calc", "optdepth_csum",
        "optdepth_depo", "optdepth_init", "optdepth_mean", "particle_init",
        "rngstate_init", "ssa_substep_1", "ssa_substep_2", "ssa_transport",
        "swarm_runtime",
    },
}

ROCM_FORBIDDEN_PATTERNS = (
    r"#include\s*<cuda",
    r"#include\s*<curand",
    r"\bCUDART_",
    r"\b__CUDA_ARCH__\b",
    r"\bcudaDeviceSynchronize\b",
    r"\bcudaGetLastError\b",
    r"\bcudaMalloc\w*\b",
    r"\bcudaMemcpy\w*\b",
    r"\bcudaMemset\w*\b",
    r"\bcudaFree\w*\b",
    r"\bcurand_init\b",
    r"\bcurand_uniform\w*\b",
    r"\bcurand_normal\w*\b",
    r"\bcuRAND\b",
    r"(?<!hip)\bcub::",
)

PRODUCTION_NAMING_PATTERNS = {
    r"\b(?:init_zspan|_get_init_zspan|z_polar_lo|z_polar_hi|z_outer|z_inner|z_neg_hi|z_pos_lo)\b":
        "use uppercase Z for cylindrical vertical coordinates",
    r"\bgasdens\b": "use gas_dens, rhog, or sigma_g according to the stored quantity",
    r"\bouter_edge\b|\bedge_[xyz]\b": "use face for interfaces and bound for boundary booleans",
    r"\bmorton_(?:query|checksum)\s*\(": "production GPU kernels must retain their canonical 13-character names",
}

GPU_KERNEL_PATTERN = re.compile(
    r"__global__\s+(?:[A-Za-z_]\w*\s+)*([A-Za-z_]\w*)\s*\("
)


def translation_units(backend: str, branch: str) -> set[str]:
    """Return the shared and selected-backend production source stems"""

    shared = {
        path.stem for path in (PROJECT_ROOT/"src"/"comm"/branch).glob("*.cu")
    }
    suffix = ".cu" if backend == "cuda" else ".hip"
    specific = {
        path.stem for path in (PROJECT_ROOT/"src"/backend/branch).glob(f"*{suffix}")
    }
    overlap = shared & specific
    if overlap:
        raise ValueError(
            f"{backend}/{branch}: duplicate shared and backend source stems: "
            f"{', '.join(sorted(overlap))}"
        )
    return shared | specific


def check_source_inventory(errors: list[str]) -> None:
    """Require each backend to resolve exactly the canonical production inventory"""

    for backend in ("cuda", "rocm"):
        for branch, expected in EXPECTED_PRODUCTION_UNITS.items():
            try:
                actual = translation_units(backend, branch)
            except ValueError as error:
                errors.append(str(error))
                continue
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing:
                errors.append(f"{backend}/{branch}: missing units: {', '.join(missing)}")
            if extra:
                errors.append(f"{backend}/{branch}: unexpected units: {', '.join(extra)}")


def check_numerical_defaults(errors: list[str]) -> None:
    """Reject drift in constants intentionally common to both backends"""

    patterns = {
        "CFL_DYN": re.compile(r"(?:const|constexpr)\s+real\s+CFL_DYN\s*=\s*([^;]+);"),
        "TPB": re.compile(r"(?:const|constexpr)\s+int\s+TPB\s*=\s*([^;]+);"),
    }
    for branch in ("fluid", "swarm"):
        paths = {
            backend: PROJECT_ROOT/"inc"/backend/branch/"const_defs.cuh"
            for backend in ("cuda", "rocm")
        }
        for name, pattern in patterns.items():
            values = {}
            for backend, path in paths.items():
                match = pattern.search(path.read_text())
                if match is None:
                    errors.append(f"{backend}/{branch}: unable to resolve {name}")
                else:
                    values[backend] = match.group(1).strip()
            if len(values) == 2 and values["cuda"] != values["rocm"]:
                errors.append(f"{branch}: CUDA and ROCm {name} defaults differ")


def check_python(errors: list[str]) -> None:
    """Parse every tracked Python source and resolve shared configuration imports"""

    for root in (PROJECT_ROOT/"val",):
        for path in root.rglob("*.py"):
            if any(part in {"logs", "temp", "__pycache__"} for part in path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as error:
                errors.append(f"{path.relative_to(PROJECT_ROOT)}: Python syntax error: {error}")
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module != "val_config":
                    continue
                for alias in node.names:
                    if alias.name != "*" and not hasattr(val_config, alias.name):
                        errors.append(
                            f"{path.relative_to(PROJECT_ROOT)}: val_config does not define {alias.name}"
                        )
            if path.name == "run.py" and any(
                parent.name.startswith("test_") for parent in path.parents
            ):
                text = path.read_text()
                guard = text.find("sys.dont_write_bytecode = True")
                shared_imports = [
                    position for marker in ("from run_model import", "from run_chain import")
                    if (position := text.find(marker)) >= 0
                ]
                if shared_imports and (guard < 0 or guard > min(shared_imports)):
                    errors.append(
                        f"{path.relative_to(PROJECT_ROOT)}: disable bytecode before importing the shared runner"
                    )


def check_rocm_sources(errors: list[str]) -> None:
    """Reject CUDA-only APIs from files selected exclusively by the ROCm backend"""

    roots = (
        PROJECT_ROOT/"inc"/"rocm",
        PROJECT_ROOT/"src"/"rocm",
        PROJECT_ROOT/"val"/"rocm",
    )
    suffixes = {".h", ".hpp", ".cuh", ".hip", ".py", ".mk"}
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(errors="replace")
            for pattern in ROCM_FORBIDDEN_PATTERNS:
                if re.search(pattern, text):
                    errors.append(
                        f"{path.relative_to(PROJECT_ROOT)}: forbidden ROCm pattern {pattern!r}"
                    )
            if "-ffast-math" in text and "-fno-finite-math-only" not in text:
                errors.append(
                    f"{path.relative_to(PROJECT_ROOT)}: -ffast-math disables NaN/Inf semantics"
                )


def check_chain_port(errors: list[str]) -> None:
    """Keep the guarded collision-chain mathematics and validation coverage paired"""

    chain_header = PROJECT_ROOT/"inc"/"comm"/"swarm"/"_col_chain.cuh"
    cache_header = PROJECT_ROOT/"inc"/"comm"/"swarm"/"_col_cache.cuh"
    if not chain_header.is_file():
        errors.append("missing shared collision-chain header")
        return
    if not cache_header.is_file():
        errors.append("missing shared collision-neighbor cache header")
        return
    for backend in ("cuda", "rocm"):
        for name in ("_col_chain.cuh", "_col_cache.cuh"):
            duplicate = PROJECT_ROOT/"inc"/backend/"swarm"/name
            if duplicate.exists():
                errors.append(f"backend collision header should be shared: {duplicate.relative_to(PROJECT_ROOT)}")

    chain_text = chain_header.read_text()
    required_chain_tokens = (
        "#ifdef GAMEDEV_CUDA",
        "curand_uniform_double(rngstate)",
        "hiprand_uniform_double(rngstate)",
        "__shared__ curs rngstate;",
        "curs rngstate; // keep the HIP RNG object local",
    )
    for token in required_chain_tokens:
        if token not in chain_text:
            errors.append(f"shared collision-chain header is missing {token!r}")

    cache_sources = {
        "rate": (
            PROJECT_ROOT/"src"/"cuda"/"swarm"/"col_rate_calc.cu",
            PROJECT_ROOT/"src"/"rocm"/"swarm"/"col_rate_calc.hip",
        ),
        "event": (
            PROJECT_ROOT/"src"/"cuda"/"swarm"/"col_event_run.cu",
            PROJECT_ROOT/"src"/"rocm"/"swarm"/"col_event_run.hip",
        ),
    }
    for name, (cuda_path, rocm_path) in cache_sources.items():
        cuda_text = cuda_path.read_text()
        rocm_text = rocm_path.read_text()
        marker = "\n#ifdef KNN_CACHE\n\n// "
        if marker not in cuda_text or marker not in rocm_text:
            errors.append(f"missing cached-Bernoulli {name} kernel block")
            continue
        cuda_block = cuda_text[cuda_text.rfind(marker):]
        rocm_block = rocm_text[rocm_text.rfind(marker):]
        if name == "event":
            cuda_block = cuda_block.replace(
                "curand_uniform_double", "backend_uniform_double"
            ).replace(
                "    __shared__ curs rngstate;",
                "    backend_rngstate;",
            )
            rocm_block = rocm_block.replace(
                "hiprand_uniform_double", "backend_uniform_double"
            ).replace(
                "    curs rngstate; // keep the HIP RNG object local because shared objects cannot be initialized",
                "    backend_rngstate;",
            )
        if cuda_block != rocm_block:
            errors.append(f"CUDA and ROCm cached-Bernoulli {name} kernels differ")

    rocm_runtime = (PROJECT_ROOT/"src"/"rocm"/"swarm"/"swarm_runtime.hip").read_text()
    required_cache_runtime = (
        "#include <_col_cache.cuh>",
        "#if !defined(BERNOULLI) || defined(KNN_CACHE)",
        "dev_col_neighbor, dev_col_measure",
        "#if defined(COLLISION_MORTON) && !defined(KNN_CACHE)",
    )
    for token in required_cache_runtime:
        if token not in rocm_runtime:
            errors.append(f"ROCm cached-Bernoulli runtime is missing {token!r}")

    for backend in ("cuda", "rocm"):
        val_common = PROJECT_ROOT/"val"/backend/"swarm"/"test_common"
        val_host = val_common/"swarm_host.cuh"
        if not val_host.is_file() or "rand_gamma_k2" not in val_host.read_text():
            errors.append(f"{backend} validation is missing the collision-initialization header override")
        val_morton = PROJECT_ROOT/"val"/backend/"swarm"/"test_knn"/"morton"/"morton_index.cuh"
        if not val_morton.is_file() or "void morton_search" not in val_morton.read_text() \
            or "void morton_digest" not in val_morton.read_text():
            errors.append(f"{backend} validation is missing the standalone Morton-query header override")

    runtime_paths = {
        "CUDA": PROJECT_ROOT/"src"/"cuda"/"swarm"/"swarm_runtime.cu",
        "ROCm": PROJECT_ROOT/"src"/"rocm"/"swarm"/"swarm_runtime.hip",
    }
    for backend, path in runtime_paths.items():
        runtime = path.read_text()
        if runtime.count("collision timestep cannot advance the operator clock") != 2:
            errors.append(
                f"{backend} runtime must guard collision-clock progress in both integrators"
            )
        if "colstate_flag <<< NB_P, TPB >>>" not in runtime:
            errors.append(f"{backend} runtime is missing the reused-geometry particle-state guard")

    test_only_tokens = (
        "COL_CACHE_VAL", "COL_GEOM_VAL", "COL_PERF_VAL", "VAL_RUNTIME_SEED",
        "GAMEDEV_VAL_INIT_SEED", "GAMEDEV_VAL_RNG_SEED",
        "KNN_FRESH", "COLLISION_UNIT_VOLUME", "COLLISION_LINEAR_TEST",
        "PERF_PARTICLES", "defined(TEST_", "#ifdef TEST_", "#ifndef TEST_",
        "rand_gamma_k2", "void morton_search", "void morton_digest",
    )
    production_paths = []
    for root in (PROJECT_ROOT/"inc", PROJECT_ROOT/"src"):
        production_paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in {".cuh", ".cu", ".hip", ".h", ".hpp"}
        )
    for path in production_paths:
        source = path.read_text()
        for token in test_only_tokens:
            if token in source:
                errors.append(
                    f"production source contains test-only token {token!r}: "
                    f"{path.relative_to(PROJECT_ROOT)}"
                )

    models = (
        "test_colchain_2d",
        "test_colchain_frag_2d",
        "test_colchain_wedge_2d",
        "test_colchain_3d",
    )
    expected_files = {"const_defs.cuh", "flags.mk", "run.py"}
    for backend in ("cuda", "rocm"):
        common_runner = PROJECT_ROOT/"val"/backend/"swarm"/"test_common"/"run_chain.py"
        if not common_runner.is_file():
            errors.append(f"missing {common_runner.relative_to(PROJECT_ROOT)}")
        for model in models:
            model_dir = PROJECT_ROOT/"val"/backend/"swarm"/model
            actual = {
                path.name for path in model_dir.iterdir() if path.is_file()
            } if model_dir.is_dir() else set()
            missing = expected_files - actual
            if missing:
                errors.append(
                    f"{backend}/{model}: missing collision-chain validation files: "
                    f"{', '.join(sorted(missing))}"
                )


def check_naming(errors: list[str]) -> None:
    """Enforce the production spelling contract without rewriting vendored KD-tree APIs"""

    suffixes = {".cu", ".hip", ".cuh", ".h", ".hpp"}
    roots = (
        PROJECT_ROOT/"inc"/"comm",
        PROJECT_ROOT/"inc"/"cuda",
        PROJECT_ROOT/"inc"/"rocm",
        PROJECT_ROOT/"src"/"comm",
        PROJECT_ROOT/"src"/"cuda",
        PROJECT_ROOT/"src"/"rocm",
    )
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes or "kdtree" in path.parts:
                continue

            source = path.read_text(errors="replace")
            for pattern, guidance in PRODUCTION_NAMING_PATTERNS.items():
                if re.search(pattern, source):
                    errors.append(
                        f"{path.relative_to(PROJECT_ROOT)}: naming violation {pattern!r}; {guidance}"
                    )

            for kernel_name in GPU_KERNEL_PATTERN.findall(source):
                if len(kernel_name) != 13:
                    errors.append(
                        f"{path.relative_to(PROJECT_ROOT)}: GPU kernel {kernel_name!r} "
                        f"has {len(kernel_name)} characters instead of 13"
                    )


def check_metadata(errors: list[str]) -> None:
    """Check vendored licenses and keep generated artifacts below val/logs or val/temp"""

    for backend in ("cuda", "rocm"):
        license_path = PROJECT_ROOT/"inc"/backend/"swarm"/"kdtree"/"Apache-2.0.txt"
        if not license_path.is_file():
            errors.append(f"missing {license_path.relative_to(PROJECT_ROOT)}")

    generated = []
    generated_suffixes = {
        ".bin", ".csv", ".d", ".dat", ".json", ".log", ".npy", ".npz",
        ".o", ".out", ".pyc", ".txt",
    }
    generated_names = {
        "gamedev", "knn_benchmark", "knn_edge_tests", "knn_periodic_tests",
        "knn_wedge_benchmark",
    }
    for path in (PROJECT_ROOT/"val").rglob("*"):
        if any(part in {"logs", "temp"} for part in path.parts):
            continue
        if path.is_file() and (
            path.suffix in generated_suffixes
            or path.name in generated_names
            or path.name.startswith((".arch_", ".target_"))
        ):
            generated.append(path.relative_to(PROJECT_ROOT))
    if generated:
        errors.append(
            "generated validation artifacts outside val/logs and val/temp: "
            + ", ".join(str(path) for path in generated[:10])
        )


def check_val_contract(errors: list[str]) -> None:
    """Keep both native matrices and the archive comparator on one definition"""

    fluid_cases = [case for group in FLUID_GROUPS.values() for case in group]
    fluid_metrics = sum(1 if "--res" in arguments else 4 for _, arguments in fluid_cases)
    if fluid_metrics != EXPECTED_FLUID_METRICS:
        errors.append(
            f"fluid validation registry yields {fluid_metrics} metrics; expected {EXPECTED_FLUID_METRICS}"
        )
    fluid_tiers = fluid_metric_tiers(fluid_cases, 4)
    if fluid_tiers["publication"] != EXPECTED_PUBLICATION_FLUID_METRICS:
        errors.append(
            f"fluid publication tier yields {fluid_tiers['publication']} metrics; "
            f"expected {EXPECTED_PUBLICATION_FLUID_METRICS}"
        )

    swarm_models = [model for group in SWARM_GROUPS.values() for model in group]
    swarm_metrics = sum(
        0 if model == "test_knn" else (
            1 if model in SWARM_FIXED_RESOLUTION else (
                2 if model in SWARM_ENDPOINT_RESOLUTION else 4
            )
        )
        for model in swarm_models
    )
    if swarm_metrics != EXPECTED_SWARM_METRICS:
        errors.append(
            f"swarm validation registry yields {swarm_metrics} metrics; expected {EXPECTED_SWARM_METRICS}"
        )
    swarm_tiers = swarm_metric_tiers(swarm_models, [32, 64, 128, 256])
    if swarm_tiers["publication"] != EXPECTED_PUBLICATION_SWARM_METRICS:
        errors.append(
            f"swarm publication tier yields {swarm_tiers['publication']} metrics; "
            f"expected {EXPECTED_PUBLICATION_SWARM_METRICS}"
        )

    for component, names in (
        ("fluid", {model for model, _ in fluid_cases}),
        ("swarm", set(swarm_models)),
    ):
        overlay_suffixes = {".cu", ".hip", ".cuh", ".h", ".hpp", ".mk", ".py", ".sh"}
        for model in sorted(names):
            common = PROJECT_ROOT/"val"/"comm"/component/model
            if not common.is_dir():
                errors.append(f"missing common validation model definition: {common.relative_to(PROJECT_ROOT)}")
            elif model != "test_knn" and not (common/"flags.mk").is_file():
                errors.append(f"missing common validation flags: {(common/'flags.mk').relative_to(PROJECT_ROOT)}")
            overlay_files = {}
            for backend in ("cuda", "rocm"):
                native = PROJECT_ROOT/"val"/backend/component/model
                if not native.is_dir():
                    errors.append(
                        f"missing {backend} validation model overlay: {native.relative_to(PROJECT_ROOT)}"
                    )
                    continue
                if not (native/"run.py").is_file():
                    errors.append(f"missing validation wrapper: {(native/'run.py').relative_to(PROJECT_ROOT)}")
                overlay_files[backend] = {
                    f"{path.stem}.gpu" if path.suffix in {".cu", ".hip"} else path.name
                    for path in native.iterdir()
                    if path.is_file()
                    and (path.name == "Makefile" or path.suffix in overlay_suffixes)
                }
            if len(overlay_files) == 2 and overlay_files["cuda"] != overlay_files["rocm"]:
                errors.append(
                    f"{component}/{model}: CUDA and ROCm overlay file sets differ after "
                    "normalizing .cu/.hip suffixes"
                )

    forbidden_paths = ("val/out", "backend_field_comparison_z.json")
    for path in (PROJECT_ROOT/"val").rglob("*.py"):
        if "__pycache__" in path.parts or path == Path(__file__).resolve():
            continue
        text = path.read_text(errors="replace")
        for stale in forbidden_paths:
            if stale in text:
                errors.append(f"{path.relative_to(PROJECT_ROOT)}: stale validation path {stale!r}")


def check_make_resolution(errors: list[str]) -> None:
    """Ask Make to resolve representative backend and QA dependency graphs"""

    commands = (
        [
            "make", "-n", "MODEL=test_colchain_2d", "GPU_BACKEND=cuda",
            "GPU_TARGET=sm_80", "COLLISION_SEARCH=morton", "VAL_SCOPE=static",
        ],
        [
            "make", "-n", "MODEL=test_colchain_2d", "GPU_BACKEND=cuda",
            "GPU_TARGET=sm_80", "COLLISION_SEARCH=kdtree", "VAL_SCOPE=static",
        ],
        [
            "make", "-n", "MODEL=test_colchain_2d", "GPU_BACKEND=rocm",
            "GPU_TARGET=gfx942", "COLLISION_SEARCH=morton", "VAL_SCOPE=static",
        ],
        [
            "make", "-n", "MODEL=test_colchain_2d", "GPU_BACKEND=rocm",
            "GPU_TARGET=gfx942", "COLLISION_SEARCH=kdtree", "VAL_SCOPE=static",
        ],
        [
            "make", "-n", "MODEL=test_initial_3d", "GPU_BACKEND=cuda",
            "GPU_TARGET=sm_80", "RES=32", "VAL_SCOPE=initialization",
        ],
        [
            "make", "-n", "MODEL=test_initial_3d", "GPU_BACKEND=rocm",
            "GPU_TARGET=gfx942", "RES=32", "VAL_SCOPE=initialization",
        ],
    )
    for command in commands:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            errors.append(
                f"Make dependency check failed for {' '.join(command[2:])}:\n{result.stdout}"
            )

    scope_checks = (
        (
            [
                "make", "-n", "MODEL=test_x_transport_2d", "GPU_BACKEND=cuda",
                "GPU_TARGET=sm_80", "RES=32", "VAL_SCOPE=transport",
            ],
            (
                "val/logs/fluid/cuda/thread/groups/transport/test_x_transport_2d",
                "val/temp/bin/cuda/fluid/test_x_transport_2d/gamedev",
                "val/temp/obj/test_x_transport_2d/fluid/cuda/thread/fast/sm_80",
            ),
        ),
        (
            [
                "make", "-n", "MODEL=test_diffusion_2d", "GPU_BACKEND=rocm",
                "GPU_TARGET=gfx942", "RES=32", "VAL_SCOPE=diffusion",
            ],
            (
                "val/logs/swarm/rocm/groups/diffusion/test_diffusion_2d",
                "val/temp/bin/rocm/swarm/test_diffusion_2d/gamedev",
                "val/temp/obj/test_diffusion_2d/swarm/rocm/gfx942",
            ),
        ),
    )
    for command, expected_paths in scope_checks:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode != 0 or any(
            expected not in result.stdout for expected in expected_paths
        ):
            errors.append(
                f"validation scope resolution failed for {' '.join(command[2:])}:\n{result.stdout}"
            )

    knn_commands = (
        ["make", "-n", "-C", str(PROJECT_ROOT/"val"/"cuda"/"swarm"/"test_knn"), "suite", "ARCH=sm_80", "K=200"],
        [
            "make", "-n", "-C", str(PROJECT_ROOT/"val"/"rocm"/"swarm"/"test_knn"),
            "suite", "AMDGPU_TARGET=gfx942", "K=200",
        ],
    )
    for command, expected in zip(
        knn_commands,
        ("val/temp/cuda/swarm/test_knn", "val/temp/rocm/swarm/test_knn"),
    ):
        result = subprocess.run(
            command, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode != 0 or expected not in result.stdout:
            errors.append(f"KNN Make dependency check failed:\n{result.stdout}")


def main() -> None:
    errors: list[str] = []
    check_source_inventory(errors)
    check_numerical_defaults(errors)
    check_python(errors)
    check_rocm_sources(errors)
    check_chain_port(errors)
    check_naming(errors)
    check_metadata(errors)
    check_val_contract(errors)
    check_make_resolution(errors)

    if errors:
        print("Merged backend static check: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Merged backend static check: PASS")
    print("static analysis passed; native CUDA and ROCm execution remains required")


if __name__ == "__main__":
    main()
