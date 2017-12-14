# From the tensorflow/models/research/ directory
python ../train.py \
    --logtostderr \
    --pipeline_config_path=/data/ylie_app/wd_object_detection/models/model/faster_rcnn_resnet101_wd.config \
    --train_dir=/data/ylie_app/wd_object_detection/models/model/train_v3
    #--pipeline_config_path=/data/ylie_app/tf_ylie_models/research/object_detection/protos/pipeline.proto \
