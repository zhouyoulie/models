#coding:utf-8

DATASET_DIR=/data/ylie_app/wd_snapshot
TRAIN_DIR=/data/ylie_app/wd_web_snapshot/train_inception_v3
CHECKPOINT_FILE=/data/ylie_app/wd_web_snapshot/train_inception_v3

python inference_image_classifier.py \
    --alsologtostderr \
    --num_classes=3 \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_split_name=validation \
    --model_name=inception_v3
