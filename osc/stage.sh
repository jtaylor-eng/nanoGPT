# stage train/val .bin to node-local NVMe; sets DATA_DIR (nanoGPT reads data/<dataset>, so symlink)
stage_data() {
  local dst=${TMPDIR:-/tmp}/fineweb10B
  if [ ! -f $dst/.STAGED ]; then mkdir -p $dst && cp $DATA_SRC/train.bin $DATA_SRC/val.bin $dst/ && touch $dst/.STAGED; fi
  export DATA_DIR=$dst
  mkdir -p $RUNDIR/data && ln -sfn $dst $RUNDIR/data/fineweb10B
}
