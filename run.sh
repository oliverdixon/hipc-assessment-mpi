#!/bin/bash

[[ $# -lt 2 ]] && slotn=$(nproc) || slotn="$2"

echo >&2 "Distributing $1 across $slotn slots."
mpirun -n "$slotn" "$1"
