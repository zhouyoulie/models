AIC_DIR="/data/ylie_app/aic/data"
INCEPTION_CHECKPOINT="/data/ylie_app/aic/model/inception_v3.ckpt"
MODEL_DIR="/data/ylie_app/aic/model"

cd /data/ylie_app/models/im2txt

bazel-bin/im2txt/train \
  --input_file_pattern="${AIC_DIR}/train-?????-of-00256" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000  # Additional 2M steps (assuming 1M in initial training).
