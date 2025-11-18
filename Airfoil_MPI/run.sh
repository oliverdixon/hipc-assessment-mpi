#!/bin/bash

[[ $# -lt 2 ]] && slotn=$(nproc) || slotn="$2"
[[ ! $(type -P mpiexec) ]] && module load OpenMPI

echo >&2 "Distributing $1 across $slotn slots."
mpiexec -n "$slotn" "$1"
