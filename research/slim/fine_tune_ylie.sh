#coding:utf-8

DATASET_DIR=/data/ylie_app/wd_snapshot
TRAIN_DIR=/data/ylie_app/wd_web_snapshot/train_inception_v3
CHECKPOINT_FILE=/data/ylie_app/wd_web_snapshot/train_inception_v3

python train_image_classifier.py\
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=wd_snapshot \
    --batch_size=20 \
    --train_image_size=299 \
    --model_name=inception_v3
    --checkpoint_path=${CHECKPOINT_FILE} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
