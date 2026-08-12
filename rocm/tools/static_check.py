#!/usr/bin/env python3

"""Check the tracked ROCm tree without requiring a ROCm installation

This check cannot replace compilation on an AMD node.  It catches incomplete
source copying, accidental CUDA API leakage, Python syntax errors, and broken
Makefile source resolution before the tree is transferred to a cluster.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


ROCM_ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROCM_ROOT.parent
PRESERVE_MANIFEST = ROCM_ROOT/"tools"/"rocm_preserve.txt"

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

SOURCE_SUFFIXES = {".h", ".hpp", ".cuh", ".hip"}
FORBIDDEN_PATTERNS = (
    r"#include\s*<cuda",
    r"#include\s*<curand",
    r"\bCUDART_",
    r"\bHIPRT_(?:INF|PI)_F\b",
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
    r"(?<!hip)\bcub::",
)


def actual_translation_units(branch: str) -> set[str]:
    """Return production HIP stems present in the isolated tree"""

    return {path.stem for path in (ROCM_ROOT/"src"/branch).glob("*.hip")}


def check_source_inventory(errors: list[str]) -> None:
    """Check the standalone production inventory and optional CUDA-tree parity"""

    for branch in ("fluid", "swarm"):
        expected = EXPECTED_PRODUCTION_UNITS[branch]
        actual = actual_translation_units(branch)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{branch}: missing HIP translation units: {', '.join(missing)}")
        if extra:
            errors.append(f"{branch}: unexpected HIP translation units: {', '.join(extra)}")

        canonical_root = CUDA_ROOT/"src"/branch
        if canonical_root.is_dir():
            canonical = {path.stem for path in canonical_root.glob("*.cu")}
            if canonical != expected:
                errors.append(
                    f"{branch}: standalone production inventory differs from the CUDA tree"
                )

    # When the canonical CUDA tree is available locally, verify every QA
    # counterpart as an additional synchronization check.  A transferred
    # standalone ROCm tree has no CUDA parent and still retains the fixed
    # production inventory check above.
    cuda_qav = CUDA_ROOT/"qav"
    if cuda_qav.is_dir():
        for source in cuda_qav.rglob("*.cu"):
            relative = source.relative_to(cuda_qav).with_suffix(".hip")
            if not (ROCM_ROOT/"qav"/relative).is_file():
                errors.append(
                    f"qav: missing HIP counterpart for {source.relative_to(CUDA_ROOT)}"
                )

    local_cuda = sorted(ROCM_ROOT.rglob("*.cu"))
    for path in local_cuda:
        errors.append(f"unexpected CUDA translation unit: {path.relative_to(ROCM_ROOT)}")


def check_numerical_defaults(errors: list[str]) -> None:
    """Reject drift in production constants that should match both backends"""

    if not (CUDA_ROOT/"inc").is_dir():
        return
    pattern = re.compile(r"(?:const|constexpr)\s+real\s+CFL_DYN\s*=\s*([^;]+);")
    for branch in ("fluid", "swarm"):
        cuda_path = CUDA_ROOT/"inc"/branch/"const_defs.cuh"
        rocm_path = ROCM_ROOT/"inc"/branch/"const_defs.cuh"
        cuda_match = pattern.search(cuda_path.read_text())
        rocm_match = pattern.search(rocm_path.read_text())
        if cuda_match is None or rocm_match is None:
            errors.append(f"{branch}: unable to resolve production CFL_DYN")
        elif cuda_match.group(1).strip() != rocm_match.group(1).strip():
            errors.append(
                f"{branch}: ROCm CFL_DYN differs from the canonical CUDA default"
            )


def check_language_files(errors: list[str]) -> None:
    """Parse Python and reject CUDA-only spellings in buildable HIP sources"""

    for path in ROCM_ROOT.rglob("*"):
        if not path.is_file() or any(part in {"obj", "out", "bin"} for part in path.parts):
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as error:
                errors.append(f"{path.relative_to(ROCM_ROOT)}: Python syntax error: {error}")
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                errors.append(f"{path.relative_to(ROCM_ROOT)}: forbidden pattern {pattern!r}")


def check_build_and_runner_backend(errors: list[str]) -> None:
    """Reject NVIDIA compiler and diagnostic commands from active build files"""

    patterns = (r"\bNVCC\b", r"\bnvcc\b", r"\bnvidia-smi\b", r"-arch=sm_[0-9]+")
    candidates = [ROCM_ROOT/"Makefile"]
    candidates.extend((ROCM_ROOT/"qav").rglob("Makefile"))
    candidates.extend(ROCM_ROOT.rglob("flags.mk"))
    candidates.extend((ROCM_ROOT/"qav").rglob("*.py"))
    for path in candidates:
        text = path.read_text(errors="replace")
        if "--resource-usage" in text:
            errors.append(
                f"{path.relative_to(ROCM_ROOT)}: unsupported ROCm 7.2 option "
                "'--resource-usage'; use the kernel-resource-usage diagnostic"
            )
        for pattern in patterns:
            if re.search(pattern, text):
                errors.append(
                    f"{path.relative_to(ROCM_ROOT)}: active backend leak {pattern!r}"
                )

        if "-ffast-math" in text and "-fno-finite-math-only" not in text:
            errors.append(
                f"{path.relative_to(ROCM_ROOT)}: -ffast-math disables NaN/Inf semantics; "
                "remove it or follow it with -fno-finite-math-only"
            )


def check_port_metadata(errors: list[str]) -> None:
    """Check licensing and the manifest protecting hand-maintained HIP files"""

    # A nested development tree inherits the repository license from its
    # parent; a private standalone cluster copy need not reproduce that file
    # merely to run structural and numerical checks
    canonical_tree_present = (
        ROCM_ROOT.name == "rocm"
        and (CUDA_ROOT/"rocm").resolve() == ROCM_ROOT.resolve()
        and (CUDA_ROOT/"Makefile").is_file()
        and (CUDA_ROOT/"src").is_dir()
    )
    if canonical_tree_present and not (CUDA_ROOT/"LICENSE").is_file():
        errors.append("missing repository MIT LICENSE")

    apache_license = ROCM_ROOT/"inc"/"swarm"/"kdtree"/"Apache-2.0.txt"
    cuda_license = CUDA_ROOT/"inc"/"swarm"/"kdtree"/"Apache-2.0.txt"
    if not apache_license.is_file():
        errors.append("missing inc/swarm/kdtree/Apache-2.0.txt")
    elif cuda_license.is_file() and apache_license.read_bytes() != cuda_license.read_bytes():
        errors.append("ROCm Apache-2.0.txt differs from the canonical CUDA-tree copy")

    if not PRESERVE_MANIFEST.is_file():
        errors.append("missing tools/rocm_preserve.txt")
        return

    paths: list[Path] = []
    for line in PRESERVE_MANIFEST.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"invalid preserve path: {entry}")
            continue
        paths.append(relative)
        if not (ROCM_ROOT/relative).is_file():
            errors.append(f"registered hand-maintained file is missing: {entry}")

    if len(paths) != len(set(paths)):
        errors.append("duplicate path in tools/rocm_preserve.txt")

    for relative in (
        Path("inc/swarm/kdtree/common.h"),
        Path("inc/swarm/kdtree/cubit/common.h"),
    ):
        text = (ROCM_ROOT/relative).read_text(errors="replace")
        if "# define __both__ __host__ __device__" not in text:
            errors.append(f"{relative}: __both__ must be explicitly host-and-device under HIP")


def check_make_resolution(errors: list[str]) -> None:
    """Ask Make to resolve representative production and QA dependency graphs"""

    commands = (
        ["make", "-n", "-C", str(ROCM_ROOT), "MODEL=fluid_fiducial", "AMDGPU_TARGET=gfx942"],
        ["make", "-n", "-C", str(ROCM_ROOT), "MODEL=swarm_fiducial", "AMDGPU_TARGET=gfx942"],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_collision_3d", "RES=32",
            "COLLISION_SEARCH=kdtree", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_collision_3d", "RES=32",
            "COLLISION_SEARCH=morton", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_failure_2d",
            "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_lds_x",
            "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_lds_reject",
            "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_sweep_block_2d",
            "RES=512", "CFL=0.45", "SAVE=1", "OUT_TIME=12.566370614359172",
            "FLUID_SWEEP=block", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_sweep_block_3d",
            "RES=64", "CFL=0.45", "SAVE=1", "OUT_TIME=10.0",
            "FLUID_SWEEP=block", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_failure_knn",
            "COLLISION_SEARCH=kdtree", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_failure_knn",
            "COLLISION_SEARCH=morton", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_perf_collision_2d",
            "PARTICLES=100000", "COLLISION_SEARCH=kdtree", "SAVE=1",
            "OUT_TIME=0.002", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT), "MODEL=test_perf_collision_2d",
            "PARTICLES=100000", "COLLISION_SEARCH=morton", "SAVE=1",
            "OUT_TIME=0.002", "AMDGPU_TARGET=gfx942",
        ],
        [
            "make", "-n", "-C", str(ROCM_ROOT/"qav"/"swarm"/"test_knn"),
            "suite", "AMDGPU_TARGET=gfx942", "K=200",
        ],
    )
    for command in commands:
        result = subprocess.run(
            command, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            errors.append(
                f"Make dependency check failed for {' '.join(command[3:])}:\n{result.stdout}"
            )


def main() -> None:
    errors: list[str] = []
    check_source_inventory(errors)
    check_numerical_defaults(errors)
    check_language_files(errors)
    check_build_and_runner_backend(errors)
    check_port_metadata(errors)
    check_make_resolution(errors)

    if errors:
        print("ROCm static check: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("ROCm static check: PASS")
    print("static analysis passed; native execution evidence is recorded separately by qav manifests")


if __name__ == "__main__":
    main()
