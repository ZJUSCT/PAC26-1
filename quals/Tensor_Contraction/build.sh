#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAC_HPCKIT_SETVARS="${PAC_HPCKIT_SETVARS:-/fs_real_a800/PAC2026/HPCKit26/HPCKit/latest/setvars.sh}"
SOURCE="${1:-${SCRIPT_DIR}/tensor_contraction.cpp}"
OUTPUT="${2:-${SCRIPT_DIR}/tensor_contraction}"

if [[ ! -f "${PAC_HPCKIT_SETVARS}" ]]; then
  echo "HPCKit environment script not found: ${PAC_HPCKIT_SETVARS}" >&2
  exit 1
fi

if [[ ! -f "${SOURCE}" ]]; then
  echo "Source file not found: ${SOURCE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set --
source "${PAC_HPCKIT_SETVARS}"
set -u

mpic++ \
  -O3 \
  -march=armv9-a+sme+sme-f64f64+sve2 \
  -std=c++17 \
  -fopenmp \
  -pthread \
  -rtlib=compiler-rt \
  -unwindlib=libunwind \
  "${SOURCE}" \
  -lnuma \
  -o "${OUTPUT}"

echo "Built ${OUTPUT}"
