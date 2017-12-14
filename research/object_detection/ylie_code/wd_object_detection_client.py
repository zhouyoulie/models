# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.append("..")

from utils import label_map_util
from grpc.beta import implementations
import tensorflow as tf
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.saved_model import signature_constants
from PIL import Image
import numpy as np
import os


tf.app.flags.DEFINE_string('server', '0.0.0.0:6007', 'PredictionService host:port')
tf.app.flags.DEFINE_string('output_save_path', '/data/ylie_app/wd_object_detection/data/save_image_1', 'path to save box imgae')
#tf.app.flags.DEFINE_string('input_path', '/data/ylie_app/wd_object_detection/data/test_data_set/guoxin_test_1.png', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('input_path', '/data/tmp/sim/host_lists', 'path to image in JPEG format')

FLAGS = tf.app.flags.FLAGS

PATH_TO_LABELS = '/data/ylie_app/wd_object_detection/data/wd_label.pbtxt'
NUM_CLASSES = 7

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def do_request(request, stub, image_file):
  image = Image.open(image_file)
  if image.mode=="RGBA":
    image = image.convert("RGB")

  
  im_width, im_height = image.size

  image_np = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image_np, axis=0)

  results = {}
  request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image_np_expanded, dtype=tf.uint8))
      #tf.contrib.util.make_tensor_proto(image_np_expanded, shape=[1, 1116, 1536, 3], dtype=tf.uint8))
  _t1 = time.time()
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  _t2 = time.time()
  logging.info("[predict] duration: %f" % (_t2-_t1))

  for key in result.outputs:
    tensor_proto = result.outputs[key]
    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
    results[key] = nd_array

  _detection_scores = results["detection_scores"][0]
  _detection_boxes = results["detection_boxes"][0]
  _detection_classes = results["detection_classes"][0]

  assert(len(_detection_boxes)==len(_detection_scores)==len(_detection_classes))
  for idx in xrange(len(_detection_boxes)):
    _score = _detection_scores[idx]
    _box = _detection_boxes[idx]
    _class = _detection_classes[idx]
    if float(_score)<=float(0.9993):
      break

    _class_id = int(_class)
    _left = int(float(_box[1]) * float(im_width))
    _right = int(float(_box[3]) * float(im_width))

    _upper = int(float(_box[0]) * float(im_height))
    _down = int(float(_box[2]) * float(im_height))

    _base_file = os.path.basename(image_file)
    _crop_box_file = os.path.join(FLAGS.output_save_path, "%s_class_%s_idx_%d.png" % (_base_file[:-4], \
      category_index[_class_id]["name"], idx))
    _crop_img = image.crop((_left, _upper, _right, _down))
    _crop_img.save(_crop_box_file)

    logging.info("[Detection Loop] image: %s, box: %r, score: %r, class: %d, box file: %s, idx: %d" % \
      (image_file, _box, _score, _class_id, _crop_box_file, idx))


def main(_):
  TEST_IMAGE_PATHS = tf.gfile.Glob(os.path.join(FLAGS.input_path, "*", "*.png"))

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inference_graph'
  request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

  _idx = 0
  for image_path_file in TEST_IMAGE_PATHS:
    _idx+=1
    _t1 = time.time()
    do_request(request, stub, image_path_file)
    _t2 = time.time()
    logging.info("[request predict] image_file: %s, idx: %d, duration: %f" % (image_path_file, _idx, (_t2-_t1)))
  

if __name__ == '__main__':
  tf.app.run()
