CHECKPOINT_PATH="/data/ylie_app/aic/model/train"
VOCAB_FILE="/data/ylie_app/aic/data/word_counts.txt"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_michael-jordan-basketball-sport-wallpapers-hd-wallpapers-hd-celebrities-sports-photo-michael-jordan-wallpaper.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_michael-jordan-the-shrug.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_Lion_vs_buffalo_3_2015-02-13.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_Michael-Jordan-main_tcm25-15662.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_cute-dog-holding-a-basketball-with-his-tongue-hanging-out-S0H934.jpg"
#IMAGE_FILE="/data/ylie_app/im2txt/data/mscoco/raw-data/val2014/test_ylie_king_lion_standing.jpg"
#IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_validation_images/fffce0267da286a3317317900dbe28f5d9a77610.jpg"

#IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/ffe30a262d3ed48129c2dd77e30df95b76d8603c.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/7ca9677e6a996b73a95df8236a8dc99338033cc1.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_train_images/8018c0ed1c8972dd0e82ab3be14985db36798877.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_train_images/fff53ca0635fd975dc827cd4efe5f79172ca8295.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_train_images/fff5195a33f64ff068fce95a8adfaa8656671cb1.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/7ffe09f636a8635a6c29009ef13b457035f79fb8.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/7ff9166080601f694758d64a970d60d38250dac1.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/7ff05315f2b08a660a67bc9c221454cb1fa9b1f4.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/fdf159e608f484972f385fa05c5907e9740e2362.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/fdf13729d1308cb378cae26d02d10d7af86e130a.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/fdcea6573f6348d6267f247e6a155cf7254cf847.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/fce41b132803cbc1587bd77df376736a7fdbec32.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/ed5d733c3fa75d7cacb2d4687ce981e278ecb341.jpg"
IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/531c76ede04dec72543a304c2135a6726090f527.jpg"
#IMAGE_FILE="/data/ylie_app/aic/data/raw-data/caption_test_images/test/*"

cd /data/ylie_app/tf_ylie_models/research/im2txt
bazel build -c opt //im2txt:run_inference

export CUDA_VISIBLE_DEVICES=""

bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}
