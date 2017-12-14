# From the tensorflow/models/research/ directory
python ../train.py \
    --logtostderr \
    --pipeline_config_path=/data/ylie_app/tf_ylie_models/research/object_detection/samples/configs/faster_rcnn_resnet101_voc07.config \
    --train_dir=/data/ylie_app/tf_ylie_models/research/object_detection/models/model/train
    #--pipeline_config_path=/data/ylie_app/tf_ylie_models/research/object_detection/protos/pipeline.proto \
