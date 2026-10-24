#!/usr/bin/env bash
set -euo pipefail
# 38 cores of NUMA node 0. OpenMP workers sleep between regions so the
# operator's own spinning worker team owns these cores during each call.
# Inputs and outputs are placed in the die-local HBM (node 16) while the
# official driver generates them; nodes without it fall back to DDR node 0.
unset GOMP_CPU_AFFINITY
mem_policy=(-m 0)
if [[ -z "${MQA_MEM_NODE:-}" && -d /sys/devices/system/node/node16 ]]; then
    mem_policy=(--preferred=16)
elif [[ -n "${MQA_MEM_NODE:-}" ]]; then
    mem_policy=(-m "$MQA_MEM_NODE")
fi
OMP_NUM_THREADS=38 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    taskset -c 0-37 numactl "${mem_policy[@]}" ./main
