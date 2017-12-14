#coding:utf-8
#########################################
# Description: Transform RectLabel object info to the mode we use 
# Author: John Zhou
#########################################

import os
import json

_cur_path = "/data/ylie_app/wd_object_detection/data/image_data_zhengjuan_v2/annotations"

_pw = open("_tag_info_file.txt", "wr+")

_all_tag_info_list = []
for each_tag_file in os.listdir(_cur_path):
  if not each_tag_file.endswith(".json"):
    continue 
  _cur_file_dict = {}
  _cur_file = os.path.join(_cur_path, each_tag_file)
  print "=" * 20
  print _cur_file
  print "=" * 20
  _cur_orig_dict = json.loads(open(_cur_file, "r").read())
  _cur_file_dict["img_key"] = _cur_orig_dict["filename"]
  _tag_list = []
  _orig_objects_list = _cur_orig_dict["objects"]
  for each_object_dict in _orig_objects_list:
    _new_tag_dict = {}
    _new_tag_dict["mark_type"] = each_object_dict["label"].encode("utf-8")
    _new_tag_dict["min_x"] = str(each_object_dict["x_y_w_h"][0])
    _new_tag_dict["min_y"] = str(each_object_dict["x_y_w_h"][1])
    _new_tag_dict["max_x"] = str(each_object_dict["x_y_w_h"][0]+each_object_dict["x_y_w_h"][2])
    _new_tag_dict["max_y"] = str(each_object_dict["x_y_w_h"][1]+each_object_dict["x_y_w_h"][3])
    _tag_list.append(_new_tag_dict)
  _cur_file_dict["tag_info"] = _tag_list
  _all_tag_info_list.append(json.dumps(_cur_file_dict))

#_str_all_tag_info = json.dumps(_all_tag_info_list)

_pw.write("\n".join(_all_tag_info_list))
_pw.close()
#print _all_tag_info_list
