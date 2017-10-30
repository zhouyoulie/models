#coding:utf-8

DATASET_DIR=/data/ylie_app/wd_snapshot
CHECKPOINT_FILE=/data/ylie_app/wd_web_snapshot/train_inception_v3

python eval_image_classifier.py \
    --alsologtostderr \
    --batch_size=5 \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=wd_snapshot \
    --dataset_split_name=validation \
    --model_name=inception_v3
