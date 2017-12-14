import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

import logging
logging.basicConfig(level=logging.INFO)

PATH_TO_CKPT = '/data/ylie_app/wd_object_detection/models/model/inference_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = '/data/ylie_app/wd_object_detection/data/wd_label.pbtxt'

NUM_CLASSES = 7

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


logging.info("[Test Detection Label] label map: %r, categories: %r, category_index: %r" % \
        (label_map, categories, category_index))

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.


PATH_TO_TEST_IMAGES_DIR = '/data/ylie_app/wd_object_detection/data/test_data_set'
#PATH_TO_TEST_IMAGES_DIR = '/data/tmp/sim/host_lists/d2b'
#PATH_TO_TEST_IMAGES_DIR = '/data/tmp/sim/host_lists/d2e'
#PATH_TO_TEST_IMAGES_DIR = '/data/tmp/sim/host_lists/d2c'
PATH_TO_TEST_IMAGES_DIR = '/data/tmp/sim/host_lists'
PATH_TO_SAVE_TEST_IMAGES_DIR = '/data/ylie_app/wd_object_detection/data/save_image'
TEST_IMAGE_PATHS = tf.gfile.Glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*", "*.png"))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    _idx = 0
    _t_start = time.time()
    for image_path in TEST_IMAGE_PATHS:
      logging.info("[Test Detection] image: %s" % image_path)
      _t1 = time.time()
      image = Image.open(image_path)
      if image.mode=="RGBA":
        image = image.convert("RGB")
      _t2 = time.time()
      logging.info("[Test Detection] image open: %f" % (_t2-_t1))
      _width = image.size[0]
      _height = image.size[1]
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      _t3 = time.time()
      image_np = load_image_into_numpy_array(image)
      _t4 = time.time()
      logging.info("[Test Detection] load_image_into_numpy_array: %f" % (_t4-_t3))
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      _t5 = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      _t6 = time.time()
      logging.info("[Test Detection] run inference: %f" % (_t6-_t5))
      #logging.info("[Test Detection Result] image: %s, boxes: %r, scores: %r, classes: %r, num: %r" % \
      #  (image_path, type(boxes), type(scores), type(classes), num))
      #logging.info("[Test Detection Result] image: %s, boxes: %r, scores: %r, classes: %r, num: %r" % \
      #  (image_path, boxes, scores, classes, num))

     
      assert(len(boxes[0])==len(scores[0])==len(classes[0])) 
      for idx in xrange(len(boxes[0])):
        _score = scores[0][idx]
        _box = boxes[0][idx]
        _class = classes[0][idx]
        if float(_score)<=float(0.9993):
          break

        _class_id = int(_class)
        _left = int(float(_box[1]) * float(_width))
        _right = int(float(_box[3]) * float(_width))

        _upper = int(float(_box[0]) * float(_height))
        _down = int(float(_box[2]) * float(_height))

        _base_file = os.path.basename(image_path)
        _crop_box_file = os.path.join(PATH_TO_SAVE_TEST_IMAGES_DIR, "%s_class_%s_idx_%d.png" % (_base_file[:-4], \
          category_index[_class_id]["name"], idx)) 
        _crop_img = image.crop((_left, _upper, _right, _down)) 
        _crop_img.save(_crop_box_file)

        logging.info("[Test Detection Loop] image: %s, box: %r, score: %r, class: %d, idx: %d" % \
          (image_path, _box, _score, _class_id, idx))
      
      _idx += 1
    _t_end = time.time()    
    logging.info("[Test Detection Loop] all duration: %f, image num: %d" % ((_t_end-_t_start), _idx))

      #print "=" * 20
      #print boxes
      #print scores
      #print classes
      #print num
      #print "=" * 20

      ## Visualization of the results of a detection.
      #vis_util.visualize_boxes_and_labels_on_image_array(
      #    image_np,
      #    np.squeeze(boxes),
      #    np.squeeze(classes).astype(np.int32),
      #    np.squeeze(scores),
      #    category_index,
      #    use_normalized_coordinates=True,
      #    line_thickness=8)
      ##plt.figure(figsize=IMAGE_SIZE)
      ##plt.imshow(image_np)
