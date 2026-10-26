#!/usr/bin/env bash
# Experimental preset; CPU resources detected at runtime.
# Pass a built ART or ART-cli executable followed by its normal arguments.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    printf 'Usage: %s /path/to/ART[-cli] [arguments...]\n' "$0" >&2
    exit 2
fi
# Preserve explicit OMP_NUM_THREADS; otherwise use detected physical cores,
# allowed affinity/cgroup quota and leave headroom for interactive work.
export ART_SCHEDULER="${ART_SCHEDULER:-dynamic}"
export ART_ASYNC_OPEN="${ART_ASYNC_OPEN:-1}"
export ART_PROGRESSIVE_OPEN="${ART_PROGRESSIVE_OPEN:-1}"
export ART_PREVIEW_CACHE="${ART_PREVIEW_CACHE:-1}"
export ART_PREFETCH_PREVIEW="${ART_PREFETCH_PREVIEW:-0}"
export ART_RAW_PARALLEL="${ART_RAW_PARALLEL:-1}"
export ART_RAW_SIMD="${ART_RAW_SIMD:-0}"
export ART_HW_MONITOR="${ART_HW_MONITOR:-1}"
export ART_CPU_OPT="${ART_CPU_OPT:-1}"
export ART_CPU_THREADS="${ART_CPU_THREADS:-1}"
export ART_EARLY_IMAGE="${ART_EARLY_IMAGE:-1}"
export ART_DENOISE_ORDERED="${ART_DENOISE_ORDERED:-1}"
export ART_CUDA="${ART_CUDA:-1}"
export ART_CUDA_RECURSIVE="${ART_CUDA_RECURSIVE:-1}"
export ART_HYBRID="${ART_HYBRID:-1}"
# ART_HYBRID_WORKERS may override automatic GPU staging-team sizing.
export ART_GPU_PAIR="${ART_GPU_PAIR:-0}"
export ART_CUDA_PINNED="${ART_CUDA_PINNED:-0}"
export ART_FFTW_ESTIMATE="${ART_FFTW_ESTIMATE:-0}"
exec "$@"
