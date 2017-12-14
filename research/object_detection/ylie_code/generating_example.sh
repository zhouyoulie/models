# generate train data
echo "[ylie]generate example"
python ylie_generate_example.py \
    --output_tag_path=/data/ylie_app/wd_object_detection/data/tmp_image_tag_info_generated_v2.txt \
    --image_data_generated=/data/ylie_app/wd_object_detection/data/image_data_generated_v2 \
    --image_data_pvuv=/data/ylie_app/wd_object_detection/data/image_data_pvuv \
    --image_data_zhengjuan=/data/ylie_app/wd_object_detection/data/image_data_zhengjuan_v2 \
    --input_tag_info_file=/data/ylie_app/wd_object_detection/data/tmp_image_tag_info_zhengjuan_v2.txt \
    --label_map_path=/data/ylie_app/wd_object_detection/data/wd_label.pbtxt 

#echo "[ylie]generate val data"
## generate val data
#python create_pascal_tf_record.py \
#    --label_map_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_label_map.pbtxt \
#    --data_dir=/data/ylie_app/pascal_object_detection/data/VOCdevkit --year=VOC2012 --set=val \
#    --output_path=/data/ylie_app/tf_ylie_models/research/object_detection/data/pascal_val.record
