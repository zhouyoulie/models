# Description: Create tf record of ylie's own data
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
flags.DEFINE_string('output_path', '', '/data/ylie_app/wd_object_detection/data/tf_record/wd_zhengjuan.record')
flags.DEFINE_string('image_data', '', 'Path to input png image data')
flags.DEFINE_string('tag_info_file', '', 'Path of tag info file')
flags.DEFINE_string('label_map_path', '', 'Path of label map file')
FLAGS = flags.FLAGS


def collect_taginfo_labelmap():
  _label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path) 

  _ret_tag_info_dict = {}
  with open(FLAGS.tag_info_file, "r") as pr:
    for each_tag_info in pr:
      #print "=" * 20
      #print each_tag_info
      #print type(each_tag_info)
      #print "=" * 20
      _tag_info_dict = json.loads(each_tag_info) 
      _str_img_key = _tag_info_dict.get("img_key", "")
      #_str_url = _str_img_key.split("?")[1].split("&")[0][len("url="):]
      #_str_datetime = _str_img_key.split("?")[1].split("&")[1][len("datetime="):]
      #_example_name = hashlib.md5(_str_url + _str_datetime).hexdigest()

      _tag_info_list = _tag_info_dict.get("tag_info", [])
      _ret_tag_info_dict[_str_img_key] = _tag_info_list
  return _label_map_dict, _ret_tag_info_dict 


def collect_image_examples():
  #with tf.gfile.GFile(FLAGS.image_data) as fid:
  #  lines = fid.readlines()
  lines = tf.gfile.Glob(os.path.join(FLAGS.image_data, "*.png"))
  #logging.info("[collect_image_examples] image_data: %r, lines: %r" % (FLAGS.image_data, lines))

  _ret_name_data_dict = {}
  height = 0
  weight = 0

  for example in lines:
    height, weight = Image.open(example).size[0], Image.open(example).size[1]

    with tf.gfile.GFile(example, 'rb') as fid:
      encoded_png_bytes = fid.read()
      _ret_name_data_dict[example] = encoded_png_bytes
  return height, weight, _ret_name_data_dict
      

def create_tf_example(filename, tag_info_list, label_map_dict):
  # TODO(user): Populate the following variables from your example.

  #height = None # Image height
  #width = None # Image width
  #filename = None # Filename of the image. Empty if image is not from file
  #encoded_image_data = None # Encoded image bytes
  #image_format = None # b'jpeg' or b'png'

  _str_image_filepath = os.path.join(FLAGS.image_data, filename)
  #logging.info("[collect_image_examples] image_data: %r, lines: %r" % (FLAGS.image_data, lines))

  _ret_name_data_dict = {}
  width, height = Image.open(_str_image_filepath).size[0], Image.open(_str_image_filepath).size[1]
  encoded_png_bytes = None

  with tf.gfile.GFile(_str_image_filepath, 'rb') as fid:
    encoded_png_bytes = fid.read()
  assert(encoded_png_bytes!=None)

  logging.info("[create_tf_example] image filename: %s, height: %d, width: %d" % (filename, height, width))

  image_format = b'png' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for _tag_dict in tag_info_list:
    _str_mark_type = _tag_dict.get("mark_type", "")
    classes_text.append(_str_mark_type.encode("utf-8"))
    _mark_type_id = label_map_dict.get(_str_mark_type, 0) 
    classes.append(_mark_type_id)
    assert(_mark_type_id!=0)
    _min_x = float(_tag_dict.get("min_x", None))
    assert(_min_x!=None)
    _max_x = float(_tag_dict.get("max_x", None))
    assert(_max_x!=None)
    _min_y = float(_tag_dict.get("min_y", None))
    assert(_min_y!=None)
    _max_y = float(_tag_dict.get("max_y", None))
    assert(_max_y!=None)
    xmins.append(_min_x/float(width))
    xmaxs.append(_max_x/float(width))
    ymins.append(_min_y/float(height))
    ymaxs.append(_max_y/float(height))

  logging.info("[create_tf_example] image filename: %s, xmin: %r, xmax: %r, ymin: %r, ymax: %r, text: %r, label: %r" % \
    (filename, xmins, xmaxs, ymins, ymaxs, classes_text, classes))
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_png_bytes),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
 
  logging.info("[main] start the program.")
  logging.info("[main] start to collect taginfo.")
  _label_map_dict, _tag_info_dict = collect_taginfo_labelmap()
  #logging.info("[main] start to collect image example.")
  #width, height, _examplename_data_dict = collect_image_examples()
  
  #logging.info("[main.info1] label_map_dict: %r, tag_info_dict: %r" % (_label_map_dict, _tag_info_dict))
  #logging.info("[main.info2] height: %r, width: %r" % (height, width))
  #logging.info("[main.info3] examplename_data_dict len: %d" % len(_examplename_data_dict))

  _idx = 0
  #for example_name, example_data in _examplename_data_dict.items():
  for filename, tag_info_list in _tag_info_dict.items():
    filename = filename.encode("utf-8")
    _idx+=1
    #tf_example = create_tf_example(example_name, example_data, height, width, _label_map_dict, _tag_info_dict)
    tf_example = create_tf_example(filename, tag_info_list, _label_map_dict)
    writer.write(tf_example.SerializeToString())
    logging.info("[main] write data idx: %d" % _idx)
  writer.close()

if __name__ == '__main__':
  tf.app.run()
