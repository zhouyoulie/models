# From the tensorflow/models/research/ directory
# for train data
python ylie_create_kb_tf_record.py \
    --logtostderr \
    --output_path=/data/ylie_app/wd_object_detection/data/tf_record/wd_zhengjuan_train.record \
    --image_data=/data/ylie_app/wd_object_detection/data/image_data_allset_v2 \
    --tag_info_file=/data/ylie_app/wd_object_detection/data/train_tag_info_list_v2.txt \
    --label_map_path=/data/ylie_app/wd_object_detection/data/wd_label.pbtxt

# for val data
python ylie_create_kb_tf_record.py \
    --logtostderr \
    --output_path=/data/ylie_app/wd_object_detection/data/tf_record/wd_zhengjuan_val.record \
    --image_data=/data/ylie_app/wd_object_detection/data/image_data_allset_v2 \
    --tag_info_file=/data/ylie_app/wd_object_detection/data/val_tag_info_list_v2.txt \
    --label_map_path=/data/ylie_app/wd_object_detection/data/wd_label.pbtxt


