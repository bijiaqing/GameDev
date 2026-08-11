#!/usr/bin/env python3

"""Refresh the tracked HIP source tree from the current CUDA implementation

The CUDA implementation remains authoritative and untouched.  This script copies
only source, configuration, documentation, and QA driver files into the top-level
ROCm tree, renames translation units from ``.cu`` to ``.hip``, and applies
the mechanical CUDA Runtime, cuRAND, and CUB spellings that have direct HIP
counterparts. ROCm-specific build and runtime decisions are maintained as
ordinary files in this tree. Paths listed in ``tools/rocm_preserve.txt`` are
snapshotted before the generated roots are replaced and restored afterward.

Run this script only when intentionally refreshing the experimental port: it
replaces the generated ``inc/``, ``src/``, ``mod/``, and ``qav/`` directories
while preserving every registered hand-maintained file.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROCM_ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROCM_ROOT.parent
PRESERVE_MANIFEST = ROCM_ROOT/"tools"/"rocm_preserve.txt"

GENERATED_ROOTS = ("inc", "src", "mod", "qav")
SKIP_DIRECTORIES = {"out", "obj", "bin", "__pycache__"}
COPY_SUFFIXES = {
    ".cu", ".cuh", ".h", ".hpp", ".py", ".mk", ".md", ".txt",
}
COPY_NAMES = {"Makefile"}


TOKEN_REPLACEMENTS = (
    ("CUDART_INF_F", "MORTON_INF_F"),
    ("CUDART_PI_F", "MORTON_PI_F"),
    ("__CUDA_ARCH__", "__HIP_DEVICE_COMPILE__"),
    ("__CUDACC__", "__HIPCC__"),
    ("_morton_cuda_check", "_morton_hip_check"),
    ("CUDA_SYNC_CHECK_STREAM", "HIP_SYNC_CHECK_STREAM"),
    ("CUDA_CHECK2_NOTHROW", "HIP_CHECK2_NOTHROW"),
    ("CUDA_CHECK_NOTHROW", "HIP_CHECK_NOTHROW"),
    ("CUDA_CALL_NOTHROW", "HIP_CALL_NOTHROW"),
    ("CUDA_KERNEL_CHECK", "HIP_KERNEL_CHECK"),
    ("CUDA_SYNC_CHECK", "HIP_SYNC_CHECK"),
    ("CUDA_SYNC_TRACE", "HIP_SYNC_TRACE"),
    ("CUDA_CHECK2", "HIP_CHECK2"),
    ("CUDA_CHECK", "HIP_CHECK"),
    ("CUDA_CALL", "HIP_CALL"),
    ("cudaFuncAttributeMaxDynamicSharedMemorySize", "hipFuncAttributeMaxDynamicSharedMemorySize"),
    ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
    ("cudaEventElapsedTime", "hipEventElapsedTime"),
    ("cudaEventSynchronize", "hipEventSynchronize"),
    ("cudaStreamSynchronize", "hipStreamSynchronize"),
    ("cudaMemcpyDeviceToDevice", "hipMemcpyDeviceToDevice"),
    ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    ("cudaMemcpyDefault", "hipMemcpyDefault"),
    ("cudaFuncSetAttribute", "hipFuncSetAttribute"),
    ("cudaGetErrorString", "hipGetErrorString"),
    ("cudaGetLastError", "hipGetLastError"),
    ("cudaMallocManaged", "hipMallocManaged"),
    ("cudaMallocAsync", "hipMallocAsync"),
    ("cudaFreeAsync", "hipFreeAsync"),
    ("cudaMallocHost", "hipHostMalloc"),
    ("cudaFreeHost", "hipHostFree"),
    ("cudaEventCreate", "hipEventCreate"),
    ("cudaEventDestroy", "hipEventDestroy"),
    ("cudaEventRecord", "hipEventRecord"),
    ("cudaMemcpy", "hipMemcpy"),
    ("cudaMemset", "hipMemset"),
    ("cudaMalloc", "hipMalloc"),
    ("cudaFree", "hipFree"),
    ("cudaError_t", "hipError_t"),
    ("cudaEvent_t", "hipEvent_t"),
    ("cudaStream_t", "hipStream_t"),
    ("cudaSuccess", "hipSuccess"),
    ("cuda_status_", "hip_status_"),
    ("cuda_fail", "hip_fail"),
    ("cuda_check", "hip_check"),
    ("cuda_vec_t", "hip_vec_t"),
    ("cuda_t", "hip_t"),
    ("cuda##", "hip##"),
    ("curand_normal_double", "hiprand_normal_double"),
    ("curand_uniform_double", "hiprand_uniform_double"),
    ("curand_init", "hiprand_init"),
    ("curandState", "hiprandState"),
    ("cub::", "hipcub::"),
)


INCLUDE_REPLACEMENTS = (
    ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
    ("#include <cuda_runtime_api.h>", "#include <hip/hip_runtime_api.h>"),
    ("#include <cuda.h>", "#include <hip/hip_runtime.h>"),
    ("#include <curand_kernel.h>", "#include <hiprand/hiprand_kernel.h>"),
    ("#include <math_constants.h>", "#include <hip/hip_math_constants.h>"),
    ("#include <cub/cub.cuh>", "#include <hipcub/hipcub.hpp>"),
)


def selected(path: Path) -> bool:
    """Return whether one repository file belongs in the isolated source port"""

    if any(part in SKIP_DIRECTORIES for part in path.parts):
        return False
    if path.name == "gamedev" or path.suffix == ".pyc":
        return False
    return path.name in COPY_NAMES or path.suffix in COPY_SUFFIXES


def destination_path(relative: Path) -> Path:
    """Rename HIP translation units while preserving every directory interface"""

    if relative.suffix == ".cu":
        relative = relative.with_suffix(".hip")
    return ROCM_ROOT / relative


def transform_cpp(text: str) -> str:
    """Apply only direct CUDA-to-HIP language and library mappings"""

    for old, new in INCLUDE_REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in TOKEN_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace('.cu"', '.hip"')
    text = text.replace("[CUDA]", "[HIP]")
    text = text.replace("CUDA error", "HIP error")
    text = text.replace("CUDA", "HIP")
    return text


def transform_make(text: str) -> str:
    """Route model feature macros through the HIP compiler flag variable"""

    return (
        text.replace("NVCC +=", "HIPFLAGS +=")
        .replace(
            "-lineinfo -Xptxas=-v",
            "-gline-tables-only",
        )
    )


def copy_tree(name: str) -> None:
    """Copy one source subtree and transform its C++ and HIP contents"""

    source_root = CUDA_ROOT / name
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(CUDA_ROOT)
        if not selected(relative):
            continue

        destination = destination_path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        if destination.suffix in {".hip", ".cuh", ".h", ".hpp"}:
            text = transform_cpp(text)
        elif destination.suffix == ".mk":
            text = transform_make(text)
        destination.write_text(text)


def preserve_paths() -> list[Path]:
    """Return validated ROCm-relative paths that survive a source refresh"""

    paths: list[Path] = []
    for line in PRESERVE_MANIFEST.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue

        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"invalid preserve path: {entry}")
        if not relative.parts or relative.parts[0] not in GENERATED_ROOTS:
            raise SystemExit(f"preserve path lies outside generated roots: {entry}")
        if not (ROCM_ROOT/relative).is_file():
            raise SystemExit(f"registered hand-maintained file is missing: {entry}")
        paths.append(relative)

    if len(paths) != len(set(paths)):
        raise SystemExit("duplicate path in tools/rocm_preserve.txt")
    return paths


def main() -> None:
    """Refresh generated subtrees while restoring registered ROCm adaptations"""

    if sys.argv[1:] != ["--force-overwrite"]:
        raise SystemExit(
            "refusing to replace the finished ROCm port; rerun with --force-overwrite "
            "only when intentionally rebuilding it from the CUDA source"
        )

    preserved = {
        relative: (ROCM_ROOT/relative).read_bytes()
        for relative in preserve_paths()
    }

    for name in GENERATED_ROOTS:
        destination = ROCM_ROOT / name
        if destination.exists():
            shutil.rmtree(destination)
        copy_tree(name)

    for relative, content in preserved.items():
        destination = ROCM_ROOT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    print(f"ROCm source mirror refreshed at {ROCM_ROOT}")
    print(f"restored {len(preserved)} hand-maintained files from {PRESERVE_MANIFEST.name}")


if __name__ == "__main__":
    main()
