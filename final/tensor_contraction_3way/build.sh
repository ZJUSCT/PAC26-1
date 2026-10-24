#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mpic++ -O3 -march=armv9-a+sve2+sme -std=c++17 -fopenmp \
  tensor_contraction_3way_mpi.cpp -o tensor_contraction_3way_mpi
