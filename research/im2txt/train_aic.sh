AIC_DIR="/data/ylie_app/aic/data"
INCEPTION_CHECKPOINT="/data/ylie_app/aic/model/inception_v3.ckpt"
MODEL_DIR="/data/ylie_app/aic/model"

cd /data/ylie_app/tf_ylie_models/research/im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${AIC_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
