# From the tensorflow/models/research/ directory
# for val data
python ylie_crop_box.py \
    --logtostderr \
    --output_path=/data/ylie_app/wd_object_detection/data/image_data_crop_box \
    --input_image_path=/data/ylie_app/wd_object_detection/data/image_data_zhengjuan \
    --tag_info_file=/data/ylie_app/wd_object_detection/data/tmp_image_tag_info_zhengjuan.txt \
    --label_map_path=/data/ylie_app/wd_object_detection/data/wd_label.pbtxt
