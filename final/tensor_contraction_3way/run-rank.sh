#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?}
exec numactl --membind="$((24+rank))" ./tensor_contraction_3way_mpi "$@"
