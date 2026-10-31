#!/bin/bash
# Fetch FineWeb10B-gpt2 train shards 11..35 (2.5B tokens) into $SCRATCH/nanogpt_data/fineweb25B/ as one train.bin;
# val.bin is a symlink to the fineweb10B one (same val split everywhere). Strips the 1024-byte shard header.
# Run on a login node (compute nodes have no HF access):  bash fetch_25B.sh
set -euo pipefail
export SCRATCH=/fs/scratch/PAS2836/$USER
D=$SCRATCH/nanogpt_data/fineweb25B; mkdir -p $D/raw; cd $D
B=https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main
for i in $(seq 11 35); do
  f=$(printf "fineweb_train_%06d.bin" $i)
  [ -s raw/$f ] && continue
  curl -sfL --retry 5 -o raw/$f.part $B/$f && mv -f raw/$f.part raw/$f
done
rm -f train.bin.part
for i in $(seq 11 35); do tail -c +1025 raw/$(printf "fineweb_train_%06d.bin" $i) >> train.bin.part; done
mv train.bin.part train.bin
ln -sfn $SCRATCH/nanogpt_data/fineweb10B/val.bin val.bin
rm -rf raw
ls -l train.bin val.bin && echo FETCH_DONE
