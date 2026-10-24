#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export UCX_TLS=sm,self OMPI_MCA_btl=vader,self
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=38 OMP_PLACES=cores OMP_PROC_BIND=close
export KMP_DISABLE_MEMKIND_INIT=true KMP_TOPOLOGY_METHOD=flat
mpirun -np 8 --oversubscribe --rankfile rankfile_8x38.txt \
  -x OMP_DYNAMIC -x UCX_TLS -x OMPI_MCA_btl -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND \
  -x KMP_DISABLE_MEMKIND_INIT -x KMP_TOPOLOGY_METHOD \
  bash ./run-rank.sh "$@"
