# Description: Crop box(logo) from png.
# Author:      Zhou You
# ================================

import os
import hashlib
import logging
logging.basicConfig(level=logging.INFO)
import json
from PIL import Image

import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', '/data/ylie_app/wd_object_detection/data/image_data_crop_box')
flags.DEFINE_string('input_image_path', '', '/data/ylie_app/wd_object_detection/data/image_data_zhengjuan')
flags.DEFINE_string('tag_info_file', '', '/data/ylie_app/wd_object_detection/data/tmp_image_tag_info_zhengjuan.txt')
flags.DEFINE_string('label_map_path', '', '/data/ylie_app/wd_object_detection/data/wd_label.pbtxt')
FLAGS = flags.FLAGS

def do_crop(tag_dict):
  str_img_key=tag_dict.get("img_key", "")  
  tag_info_list=tag_dict.get("tag_info", [])  
  
  img_key_list=str_img_key.split("image/get?")
  assert(len(img_key_list)==2)
  _url_datetime_list = img_key_list[1].split("&")
  _str_url = _url_datetime_list[0][len("url="):]
  _str_datetime = _url_datetime_list[1][len("datetime="):]
  _filename=hashlib.md5(_str_url+_str_datetime).hexdigest()

  _index=0
  for single_item_dict in tag_info_list:
    _min_x = single_item_dict.get("min_x", None)
    assert(_min_x!=None)
    _min_x_coordinate = int(_min_x)

    _min_y = single_item_dict.get("min_y", None)
    assert(_min_y!=None)
    _min_y_coordinate = int(_min_y)

    _max_x = single_item_dict.get("max_x", None)
    assert(_max_x!=None)
    _max_x_coordinate = int(_max_x)

    _max_y = single_item_dict.get("max_y", None)
    assert(_max_y!=None)
    _max_y_coordinate = int(_max_y)

    _str_mark_type = single_item_dict.get("mark_type", None)
    assert(_str_mark_type!=None)
   
    _str_url = _str_url.encode("utf-8")
    _str_datetime = _str_datetime.encode("utf-8")
    _str_mark_type_encode = _str_mark_type.encode("utf-8")
    str_box_=_str_url+_str_datetime+_str_mark_type_encode

    _box_filename=hashlib.md5("%s%d"%(_str_url+_str_datetime+_str_mark_type_encode, _index)).hexdigest()
    _index+=1
    _to_open_file = os.path.join(FLAGS.input_image_path, "zhengjuan_%s.png"%_filename)
    _imag = Image.open(_to_open_file)
    _imag_box = _imag.crop((_min_x_coordinate, _min_y_coordinate, _max_x_coordinate, _max_y_coordinate))
    _imag_box.save(os.path.join(FLAGS.output_path, "box_%s.png"%_box_filename))


def run_crop():
  with open(FLAGS.tag_info_file, "r") as pr:
    for line in pr:
      str_line = line.strip()
      _line_dict = json.loads(str_line)
      do_crop(_line_dict)

def main(_):
  run_crop()

if __name__=="__main__":
  tf.app.run()
