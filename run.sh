#!/usr/bin/env bash
# run.sh — build & run the grayscale example

set -e
make clean && make all

# ensure output folder exists
mkdir -p data

# run with 16×16 threads/block, 32×32 blocks/grid
./bin/imageRotationNPP \
  --input  data/Lena.png \
  --output data/Lena_gray.png \
  --block  16 16 1 \
  --grid   32 32 1

echo "Wrote data/Lena_gray.png"
