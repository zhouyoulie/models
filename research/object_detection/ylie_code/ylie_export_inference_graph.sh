# From tensorflow/models/research/
python ../export_inference_graph.py \
    --input_type='image_tensor' \
    --pipeline_config_path=/data/ylie_app/wd_object_detection/models/model/faster_rcnn_resnet101_wd.config \
    --trained_checkpoint_prefix=/data/ylie_app/wd_object_detection/models/model/train_v3_bak/model.ckpt-65927 \
    --output_directory=/data/ylie_app/wd_object_detection/models/model/inference_graph/2
