#!/usr/bin/env bash
# Shared setup for one-GPU temporal diagnostics. Source after selecting the code root.
source scripts/slurm/_preamble.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

diagnostic_init() {
    mkdir -p "$1"
    python - "${2:-}" <<'PYCUDA'
import sys
import torch
assert torch.cuda.is_available(), "temporal diagnostics require CUDA"
if sys.argv[1]:
    assert sys.argv[1] in torch.cuda.get_device_name()
PYCUDA
}
