# Location to save the ai challenge data.
AIC_DIR="/data/ylie_app/aic/data"

# Build the preprocessing script.
cd /data/ylie_app/models/im2txt
bazel build //im2txt:preprocess_aic

# Run the preprocessing script.
bazel-bin/im2txt/preprocess_aic "${AIC_DIR}"
