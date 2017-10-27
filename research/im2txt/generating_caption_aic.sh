CHECKPOINT_PATH="/data/ylie_app/aic/model/train"
VOCAB_FILE="/data/ylie_app/aic/data/word_counts.txt"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_michael-jordan-basketball-sport-wallpapers-hd-wallpapers-hd-celebrities-sports-photo-michael-jordan-wallpaper.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_michael-jordan-the-shrug.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_Lion_vs_buffalo_3_2015-02-13.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_Michael-Jordan-main_tcm25-15662.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_cute-dog-holding-a-basketball-with-his-tongue-hanging-out-S0H934.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_king_lion_standing.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_validation_images/fffce0267da286a3317317900dbe28f5d9a77610.jpg"

cd /data/ylie_app/models/im2txt
bazel build -c opt //im2txt:run_inference

export CUDA_VISIBLE_DEVICES=""

bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}
