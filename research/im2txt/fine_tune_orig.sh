MSCOCO_DIR="/data/ylie_app/im2txt/data/mscoco"
INCEPTION_CHECKPOINT="/data/ylie_app/im2txt/data/inception_v3.ckpt"
MODEL_DIR="/data/ylie_app/im2txt/model"

cd /data/ylie_app/models/im2txt

bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=3000000  # Additional 2M steps (assuming 1M in initial training).
