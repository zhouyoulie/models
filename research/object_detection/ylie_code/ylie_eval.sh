# From the tensorflow/models/research/ directory
python ../eval.py \
    --logtostderr \
    --pipeline_config_path=/data/ylie_app/wd_object_detection/models/model/faster_rcnn_resnet101_wd.config \
    --eval_dir=/data/ylie_app/wd_object_detection/models/model/eval \
    --checkpoint_dir=/data/ylie_app/wd_object_detection/models/model/train \
    #--pipeline_config_path=/data/ylie_app/tf_ylie_models/research/object_detection/protos/pipeline.proto \
