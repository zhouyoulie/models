import json
import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_string('wd_tag_input_file', '/data/ylie_app/wd_object_detection/data/tmp_image_tag_info.txt',
                    'Original WD image tag file path v1')
flags.DEFINE_string('wd_tag_input_path', '/data/ylie_app/wd_object_detection/data/image_data_zhengjuan_v2/annotations',
                    'Original WD image tag file path v2')
flags.DEFINE_string('wd_label_map_out_file', '/data/ylie_app/wd_object_detection/data/wd_label.pbtxt', 'WD label file path')

FLAGS = flags.FLAGS

'''
item {
  id: 1
  name: 'aeroplane'
}

item {
  id: 2
  name: 'bicycle'
}
'''

def transform_v1():
  _tag_list = []
  with open(FLAGS.wd_tag_input_file, "r") as pr:
    for line in pr:
      str_line = line.strip()
      _line_dict = json.loads(str_line)
      _tag_info_list = _line_dict.get("tag_info", [])
      for _item_dict in _tag_info_list:
        _tag_list.append(_item_dict.get("mark_type", ""))

  _idx = 1
  _tag_set = set(_tag_list)
  _pw = open(FLAGS.wd_label_map_out_file, "wr+")
  for _tag in _tag_set:
      _pw.write("item {\n  id: %d\n  name: '%s'\n}\n\n" % (_idx, _tag.encode("utf-8")))
      _idx += 1
  _pw.close()



def transform_v2():
  _tag_info_json_list = [os.path.join(FLAGS.wd_tag_input_path, _file_path) for _file_path in \
    os.listdir(FLAGS.wd_tag_input_path) if os.path.isfile(os.path.join(FLAGS.wd_tag_input_path,_file_path))] 

  print os.listdir(FLAGS.wd_tag_input_path)

  _tag_object_list = []
  
  for _each_file in _tag_info_json_list:
    _tag_info_dict = json.loads(open(_each_file, "r").read().strip())
    _objects_label_list = [_label_dict["label"] for _label_dict in _tag_info_dict.get("objects")]
    _tag_object_list.extend(_objects_label_list)

  print "=" * 20
  print "%r" % _tag_info_json_list
  print "=" * 20

  _idx = 1
  _tag_set = set(_tag_object_list)
  _pw = open(FLAGS.wd_label_map_out_file, "wr+")
  for _tag in _tag_set:
      _pw.write("item {\n  id: %d\n  name: '%s'\n}\n\n" % (_idx, _tag.encode("utf-8")))
      _idx += 1
  _pw.close()


def main(_):
  #transform_v1()
  transform_v2()
  

if __name__ == '__main__':
  tf.app.run()
