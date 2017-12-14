# generate train data
echo "[ylie]generate train data"
python create_pascal_tf_record.py \
    --label_map_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/data/ylie_app/pascal_object_detection/data/VOCdevkit --year=VOC2012 --set=train \
    --output_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_train.record


echo "[ylie]generate val data"
# generate val data
python create_pascal_tf_record.py \
    --label_map_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/data/ylie_app/pascal_object_detection/data/VOCdevkit --year=VOC2012 --set=val \
    --output_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_val.record
