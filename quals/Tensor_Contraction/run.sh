#!/usr/bin/env bash
set -eo pipefail

source /fs_real_a800/PAC2026/HPCKit26/HPCKit/latest/setvars.sh
set -u

export UCX_TLS=sm,self
export OMPI_MCA_btl=vader,self
export OMP_NUM_THREADS=36
export OMP_STACKSIZE=8M
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 8 --rankfile rankfile_8x36.txt \
  -x UCX_TLS -x OMPI_MCA_btl \
  -x OMP_NUM_THREADS -x OMP_STACKSIZE -x OMP_PLACES -x OMP_PROC_BIND \
  ./tensor_contraction
