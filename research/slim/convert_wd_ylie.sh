#coding:utf-8

DATA_DIR=/data/ylie_app/wd_snapshot
python download_and_convert_data.py \
  --dataset_name=wd_web_snapshot \
  --dataset_dir="${DATA_DIR}"
