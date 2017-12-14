# Description: Create pasting image based on pvuv images
# Author:      Zhou You
# ================================

import json
import hashlib
import random
from random import randint
import numpy as np
import os
from PIL import Image, ImageEnhance
import tensorflow as tf
import time
import logging
logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('output_tag_path', '', '/data/ylie_app/wd_object_detection/data/tmp_image_tag_info_generated.txt')
flags.DEFINE_string('image_data_generated', '', 'Path to output png image data')
flags.DEFINE_string('image_data_pvuv', '', 'Path to input pvuv image data')
flags.DEFINE_string('image_data_zhengjuan', '', 'Path to input png image data')
flags.DEFINE_string('input_tag_info_file', '', 'Path of tag info file')
flags.DEFINE_string('label_map_path', '', 'Path of label map file')
FLAGS = flags.FLAGS

#def process_line_tag_info(line_tag_info_list):
#  _tag_info_dict = {}
#  assert(len(line_tag_info_list) > 0)
#  _idx = 0
#  for tag_dict in line_tag_info_list:
#    _min_x = tag_dict.get("min_x", None) 
#    assert(_min_x!=None)
#    _min_x_coordinates = int(_min_x)
#
#    _min_y = tag_dict.get("min_y", None) 
#    assert(_min_y!=None)
#    _min_y_coordinates = int(_min_y)
#
#    _max_x = tag_dict.get("max_x", None) 
#    assert(_max_x!=None)
#    _max_x_coordinates = int(_max_x)
#
#    _max_y = tag_dict.get("max_y", None) 
#    assert(_max_y!=None)
#    _max_y_coordinates = int(_max_y)
#
#    _str_mark_type = tag_dict.get("mark_type", None)
#    assert(_str_mark_type!=None)
#    _str_mark_type = _str_mark_type.encode("utf-8")
#
#    _idx += 1



class SingleBox(object):
  def __init__(self, left, upper, width, height):
    self._x1=left
    self._x2=left+width
    self._y1=upper
    self._y2=upper+height
 
  @property
  def x_1(self):
    return self._x1
 
  @property
  def x_2(self):
    return self._x2

  @property
  def y_1(self):
    return self._y1
 
  @property
  def y_2(self):
    return self._y2
 
  @property
  def left_upper_coordinate(self):
    return (self._x1, self._y1)

  @property
  def left_down_coordinate(self):
      return (self._x1, self._y2)

  @property
  def right_upper_coordinate(self):
    return (self._x2, self._y1)

  @property
  def right_down_coordinate(self):
    return (self._x2, self._y2) 

  
  def has_intersection(self, box_other):
    #b_has_intersection = False

    _min_x = max(self.x_1, box_other.x_1)
    _min_y = max(self.y_1, box_other.y_1)
    _max_x = min(self.x_2, box_other.x_2)
    _max_y = min(self.y_2, box_other.y_2)
 
    return not(_min_x >= _max_x or _min_y >= _max_y)
 
    #if self.x_1 > box_other.x_1 and self.x_1 < box_other.x_2 and\
    #   self.y_1 > box_other.y_1 and self.y_1 < box_other.y_2:
    #  b_has_intersection = True
    #elif self.x_2 > box_other.x_1 and self.x_2 < box_other.x_2 and\
    #   self.y_1 > box_other.y_1 and self.y_1 < box_other.y_2:
    #  b_has_intersection = True
    #elif self.x_1 > box_other.x_1 and self.x_1 < box_other.x_2 and\
    #   self.y_2 > box_other.y_1 and self.y_2 < box_other.y_2:
    #  b_has_intersection = True
    #elif self.x_2 > box_other.x_1 and self.x_2 < box_other.x_2 and\
    #   self.y_2 > box_other.y_1 and self.y_2 < box_other.y_2:
    #  b_has_intersection = True
    #elif self.x_1 <= box_other.x_1 and self.x_2 >= box_other.x_2 and\
    #    self.y_1 <= box_other.y_1 and self.y_2 >= box_other.y_2:
    #  b_has_intersection = True
    #
    #return b_has_intersection

class PastedBoxList(object):
  def __init__(self):
    self._single_box_list = []

  def _add_box(self, single_box):
    assert(isinstance(single_box, SingleBox))
    self._single_box_list.append(single_box) 

  def _check_if_conflict(self, single_box):
    if len(self._single_box_list) == 0:
      return False
    assert(isinstance(single_box, SingleBox))
    for item_box in self._single_box_list:
      if single_box.has_intersection(item_box):
        return True 
    return False

class BoxPasted(object):
  def __init__(self, _parent_width, _parent_height):
    self._parent_width = _parent_width
    self._parent_height = _parent_height
    self._box_list = []

  def __del__(self):
    pass

def crop_image(tag_info_list, image_base_name):
  _str_image_file = os.path.join(FLAGS.image_data_zhengjuan, image_base_name)
  _orig_img = Image.open(_str_image_file)

  _ret_list = []
  for tag_info_dict in tag_info_list:
    _new_dict = {}
    _min_x = tag_info_dict.get("min_x", None) 
    assert(_min_x!=None)
    _min_x_coordinates = int(_min_x)
    _new_dict["min_x"] = _min_x_coordinates
    
    _min_y = tag_info_dict.get("min_y", None) 
    assert(_min_y!=None)
    _min_y_coordinates = int(_min_y)
    _new_dict["min_y"] = _min_y_coordinates
    
    _max_x = tag_info_dict.get("max_x", None) 
    assert(_max_x!=None)
    _max_x_coordinates = int(_max_x)
    _new_dict["max_x"] = _max_x_coordinates
    
    _max_y = tag_info_dict.get("max_y", None) 
    assert(_max_y!=None)
    _max_y_coordinates = int(_max_y)
    _new_dict["max_y"] = _max_y_coordinates

    _str_mark_type = tag_info_dict.get("mark_type", None)
    assert(_str_mark_type!=None)
    _str_mark_type = _str_mark_type.encode("utf-8")
    _new_dict["mark_type"] = _str_mark_type

    _crop_img = _orig_img.crop((_min_x_coordinates, _min_y_coordinates, _max_x_coordinates, _max_y_coordinates)) 
    _new_dict["crop_img"] = _crop_img
    _ret_list.append(_new_dict)
  return _ret_list


def preprocess_tag_info():
  _ret_tag_info_list = []
  with open(FLAGS.input_tag_info_file) as pr:
    _idx = 1
    for line in pr:
      logging.info("[preprocess_tag_info] processing: %d" % _idx)
      _line_dict = json.loads(line.strip()) 
      img_key = _line_dict.get("img_key", "") 
      #_str_url = img_key.split("image/get?")[1].split("&")[0][len("url="):]
      #_str_datetime = img_key.split("image/get?")[1].split("&")[1][len("datetime="):]
      #_line_dict["url"] = _str_url
      #_line_dict["datetime"] = _str_datetime
      #_line_dict["image_file_base_name"] = "zhengjuan_%s.png" % hashlib.md5(_str_url+_str_datetime).hexdigest() 

      _line_dict["image_file_base_name"] = img_key

      _tag_info_list = _line_dict["tag_info"]
      _tag_info_list_with_crop_image = crop_image(_tag_info_list, _line_dict["image_file_base_name"])
      _line_dict["tag_info"] = _tag_info_list_with_crop_image
      _line_dict["index"] = _idx
      _idx+=1

      _ret_tag_info_list.append(_line_dict)
  logging.info("[preprocess_tag_info] processing tag info list len: %d, %r" % (len(_ret_tag_info_list), _ret_tag_info_list))
  return _ret_tag_info_list


def _generate_compositing_images(num_example, image_file_path, str_url, str_datetime, mark_type_list, \
      min_x_list, max_x_list, min_y_list, max_y_list):
  assert(len(min_x_list)==len(max_x_list))
  assert(len(min_y_list)==len(max_y_list))
  assert(len(min_y_list)==len(min_x_list))
  assert(len(min_y_list)==len(mark_type_list))
  logging.info("[generate_compositing_images] 1")

  _pw = open(FLAGS.output_tag_path, "a+")

  _original_img = Image.open(image_file_path)
  _original_img_width = _original_img.size[0]
  _original_img_height = _original_img.size[1]

  logging.info("[generate_compositing_images] 2")

  _images_list = tf.gfile.Glob(os.path.join(FLAGS.image_data_pvuv, "pvuv_*.png"))
  for _new_image in _images_list:
    _base_pvuv_name = os.path.basename(_new_image)
    logging.info("[generate_compositing_images] 3")
    for i in xrange(0, len(min_x_list)):
      logging.info("[generate_compositing_images] 4")
      _idx = 0
      _box_width = max_x_list[i] - min_x_list[i]
      _box_height = max_y_list[i] - min_y_list[i]
      _gap_width = _original_img_width - _box_width
      _gap_height = _original_img_width - _box_width
      _crop_img = _original_img.crop((min_x_list[i], min_y_list[i], max_x_list[i], max_y_list[i]))

      for _paste_left_x in xrange(1, int(_original_img_width-_box_width), 30):
        if _idx >= num_example: break 
        for _paste_upper_y in xrange(1, int(_original_img_height-_box_height), 30):
          if _idx >= num_example: break 
          _idx += 1
          _new_img = Image.open(_new_image)
          _new_img_resized = _new_img.resize((1536, 1116))
          _new_img_resized.paste(_crop_img, (_paste_left_x, _paste_upper_y))
          _image_filename = "pvuv_%s.png" % hashlib.md5("%s_%s_%d" % (_base_pvuv_name, mark_type_list[i].encode("utf-8"), _idx)).hexdigest()

          _paste_max_x = int(_paste_left_x+_box_width)
          assert(_paste_max_x<=_original_img_width)
          _paste_max_y = int(_paste_upper_y+_box_height)
          assert(_paste_max_y<=_original_img_height)
          _new_tag_info_dict = {"img_key": _image_filename,\
            "tag_info": [{"min_x":str(_paste_left_x),"min_y":str(_paste_upper_y),"max_x":str(_paste_max_x),\
            "max_y":str(_paste_max_y),"mark_type":mark_type_list[i]}]}
          _str_new_tag_info = json.dumps(_new_tag_info_dict)
          _pw.write("%s\n" % _str_new_tag_info)
          _new_img_resized.save(os.path.join(FLAGS.image_data_generated, _image_filename))
          logging.info("[generate_compositing_images] new file: %s" % os.path.join(FLAGS.image_data_generated, _image_filename))
  _pw.close()
  logging.info("[generate_compositing_images] 3")



def _do_paste_single_image_with_logoes(orig_image, random_selected_tag_info_list, index):
  logging.info("[do_paste_single_image_with_logoes]orig_image: %s." % orig_image)

  _orig_image_to_paste = Image.open(orig_image)
  _orig_image_to_paste = _orig_image_to_paste.resize((1536, 1116)) 
  _basename = os.path.basename(orig_image)
  _to_save_filename = "generated_%s.png" % hashlib.md5("%s%f"%(_basename, time.time())).hexdigest()
  _to_save_file_path = os.path.join(FLAGS.image_data_generated, _to_save_filename)

  _orig_width = _orig_image_to_paste.size[0]
  _orig_height = _orig_image_to_paste.size[1]

  _pasted_box_list = PastedBoxList() 
  _pw = open(FLAGS.output_tag_path, "a+")

  _tag_info_list = []
  _write_tag_info_dict = {"img_key":_to_save_filename, "tag_info":_tag_info_list} 

  _paste_num = 0
  for single_tag_info_dict in random_selected_tag_info_list:
    _tag_info_dict = {}
    _len_single_tag_info = len(single_tag_info_dict["tag_info"])
    _first_tag_info_dict = single_tag_info_dict["tag_info"][randint(0, _len_single_tag_info-1)]

    _str_mark_type = _first_tag_info_dict["mark_type"]
    _tag_info_dict["mark_type"] = _str_mark_type

    _crop_img = _first_tag_info_dict["crop_img"]
    _crop_width, _crop_height = _crop_img.size[0], _crop_img.size[1]
    if _crop_width >= 100 or _crop_height >= 100:
        _resized_percentage = float(np.random.choice(np.arange(0.3, 1.0, 0.2)))
    elif _crop_width < 100 or _crop_height < 100:
        _resized_percentage = float(np.random.choice(np.arange(1.0, 1.5, 0.2)))

    _crop_resized_height, _crop_resized_width = \
      int(float(_crop_height)*_resized_percentage), int(float(_crop_width)*_resized_percentage)

    _crop_img = _crop_img.resize((_crop_resized_width, _crop_resized_height))

    _crop_img_obj = ImageEnhance.Brightness(_crop_img)
    _brightness_enhance_ratio = float(np.random.choice(np.arange(0.7, 1.0, 0.1)))
    _crop_img_bn = _crop_img_obj.enhance(_brightness_enhance_ratio)

    _crop_img_bn_obj = ImageEnhance.Color(_crop_img_bn)
    _color_enhance_ratio = float(np.random.choice(np.arange(0.7, 1.0, 0.1)))
    _crop_img_bn_color = _crop_img_bn_obj.enhance(_color_enhance_ratio)

    _crop_img_bn_color_obj = ImageEnhance.Contrast(_crop_img_bn_color)
    _contrast_enhance_ratio = float(np.random.choice(np.arange(0.7, 1.0, 0.1)))
    _crop_img_bn_color_contrast = _crop_img_bn_color_obj.enhance(_contrast_enhance_ratio)

    _crop_img_bn_color_contrast_obj = ImageEnhance.Sharpness(_crop_img_bn_color_contrast)
    _sharpness_enhance_ratio = float(np.random.choice(np.arange(0.7, 1.0, 0.1)))
    _crop_img_bn_color_contrast_sharp = _crop_img_bn_color_contrast_obj.enhance(_sharpness_enhance_ratio)

    logging.info("[do_paste_single_image_with_logoes]to save file: %s, mark_type: %s, resized_percentage: %f, \
      brightness_enhance_ratio: %f, color_enhance_ratio: %f, contrast_enhance_ratio: %f, sharpness_enhance_ratio: %f" \
      % (_to_save_filename, _str_mark_type, _resized_percentage, _brightness_enhance_ratio, _color_enhance_ratio, \
      _contrast_enhance_ratio, _sharpness_enhance_ratio))

    #loop 5 times
    _try_index = 1
    for i in xrange(5): 
      logging.info("[do_paste_single_image_with_logoes]try index: %d" % _try_index)
      _try_index += 1
      _paste_left_x_random_pick = randint(0, _orig_width-_crop_resized_width) 
      _paste_upper_y_random_pick = randint(0, _orig_height-_crop_resized_height) 

      _single_box = SingleBox(_paste_left_x_random_pick, _paste_upper_y_random_pick, _crop_resized_width, _crop_resized_height) 
      if not _pasted_box_list._check_if_conflict(_single_box):
        _pasted_box_list._add_box(_single_box)
        _orig_image_to_paste.paste(_crop_img_bn_color_contrast_sharp, (_paste_left_x_random_pick, _paste_upper_y_random_pick))
        _paste_num += 1
        
        _tag_info_dict["min_x"] = str(_paste_left_x_random_pick)
        _tag_info_dict["max_x"] = str(_paste_left_x_random_pick+_crop_resized_width)

        _tag_info_dict["min_y"] = str(_paste_upper_y_random_pick)
        _tag_info_dict["max_y"] = str(_paste_upper_y_random_pick+_crop_resized_height)
        _tag_info_list.append(_tag_info_dict)
        break
   
  logging.info("[do_paste_single_image_with_logoes]to_save_file_path: %s, paste num: %d\n" % (_to_save_file_path, _paste_num))
  _orig_image_to_paste.save(_to_save_file_path)  
  
  _str_tag_info_dict = json.dumps(_write_tag_info_dict)
  logging.info("[do_paste_single_image_with_logoes] write tag info: %s" % _str_tag_info_dict)
  _pw.write("%s\n" % _str_tag_info_dict)

     
    #_new_img.paste(_crop_img, (_paste_left_x, _paste_upper_y))

    # how to paste 3 images
    #_orig_image.paste()
    #_new_img.paste(_crop_img, (_paste_left_x, _paste_upper_y))

  _pw.close()

def _paste_single_image_with_logoes(orig_image, tag_info_list, _duplicate_num):
  for i in xrange(_duplicate_num):
    logging.info("[paste_single_image]orig_image: %s. %dth dup of %d dups" % (orig_image, i, _duplicate_num))
    # a mini batch boxes to paste on orig_image
    _random_selected_tag_info_list=random.sample(tag_info_list, len(tag_info_list)/20)
    logging.info("[paste_single_image]_random_selected_tag_info_list , len: %d. all len: %d" % \
      (len(_random_selected_tag_info_list), len(tag_info_list)))
    _do_paste_single_image_with_logoes(orig_image, _random_selected_tag_info_list, i)
    

def _generate_images_based_on_all_tag(_tag_info_list, _duplicate_num):
  logging.info("[generate_images_based_on_all_tag] 1") 
  _images_list = tf.gfile.Glob(os.path.join(FLAGS.image_data_pvuv, "*_*.png"))

  for _single_image in _images_list:
    logging.info("[generate_images_based_on_all_tag] processing single_image: %s" % _single_image) 
    _paste_single_image_with_logoes(_single_image, _tag_info_list, _duplicate_num)


  #for _tag_info_dict in _tag_info_list:
  #  _str_image_file_base_name = _tag_info_dict.get("image_file_base_name", None) 
  #  assert(_str_image_file_base_name!=None)
  

## for each tag, we generate constant num image samples
#def _generate_each_tag_images(tag_dict):
#  logging.info("[generate_each_tag_images] 1")
#  _str_img_key = tag_dict.get("img_key", "")
#  _str_url = _str_img_key.split("?")[1].split("&")[0][len("url="):]
#  _str_datetime = _str_img_key.split("?")[1].split("&")[1][len("datetime="):]
#  _example_name = hashlib.md5(_str_url + _str_datetime).hexdigest()
#
#  _mark_type_list = []
#  _min_x_list = []
#  _max_x_list = []
#  _min_y_list = []
#  _max_y_list = []
#
#  tag_list = tag_dict.get("tag_info", [])
#  logging.info("[generate_each_tag_images] 2")
#  for _tag_dict in tag_list:
#    _str_mark_type = _tag_dict.get("mark_type", "")
#    _mark_type_list.append(_str_mark_type)
#    _min_x = float(_tag_dict.get("min_x", None))
#    _min_x_list.append(_min_x)
#    _max_x = float(_tag_dict.get("max_x", None))
#    _max_x_list.append(_max_x)
#    _min_y = float(_tag_dict.get("min_y", None))
#    _min_y_list.append(_min_y)
#    _max_y = float(_tag_dict.get("max_y", None))
#    _max_y_list.append(_max_y)
#
#
#  #_images_list = tf.gfile.Glob(os.path.join(FLAGS.image_data, "zhengjuan_*.png"))
#  _image_file = os.path.join(FLAGS.image_data, "zhengjuan_%s.png" % _example_name)
#
#  logging.info("[generate_each_tag_images] 3")
#
#  #for _image_file in _images_list:
#  logging.info("[generate_each_tag_images] 4")
#  _num = 400
#  _generate_compositing_images(_num, _image_file, _str_url, _str_datetime, _mark_type_list, \
#    _min_x_list, _max_x_list, _min_y_list, _max_y_list)


def main(_):
  _tag_info_list = preprocess_tag_info()

  _duplicate_num = 500
  _generate_images_based_on_all_tag(_tag_info_list, _duplicate_num)

  #with open(FLAGS.input_tag_info_file) as pr:
  #  _idx = 0
  #  for line in pr:
  #    logging.info("[main] looping tag %d" % _idx)
  #    _line_dict = json.loads(line.strip())
  #    logging.info("[main] looping tag line_dict: %r" % _line_dict)
  #    _generate_each_tag_images(_line_dict)
  #    _idx += 1

if __name__ == '__main__':
  tf.app.run()
