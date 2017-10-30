# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
from shutil import copyfile

from datasets import dataset_factory
from nets import nets_factory
from nets import inception
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'num_classes', 0,
    'An number for the labels in the dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'inference_image_size', None, 'Predict image size')

FLAGS = tf.app.flags.FLAGS


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    pass

  def read_image_dims(self, image_data):
    image = self.decode_png(image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, image_data):
    image = tf.image.decode_png(image_data, channels=3)
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
 
def _check_bad_case_dir(class_names, dataset_dir):
  wd_snapshot_root = os.path.join(dataset_dir, 'wd_snapshot_photo')
  _str_bad_case_dir = os.path.join(wd_snapshot_root, "bad_case_dir")

  if not tf.gfile.Exists(_str_bad_case_dir):
    tf.gfile.MakeDirs(_str_bad_case_dir)

  for item_class_name in class_names:
    _str_bad_case_sub_dir = os.path.join(_str_bad_case_dir, item_class_name)
    if not tf.gfile.Exists(_str_bad_case_sub_dir):
      tf.gfile.MakeDirs(_str_bad_case_sub_dir)
  #black, white, unknown sub directories in ret dir perspectively
  return _str_bad_case_dir

def _action_if_predict_wrong(prediction, filename, class_names_to_ids, class_ids_to_names, str_bad_case_dir):
  _predict_id = prediction[0]
  _str_wrong_names = class_ids_to_names[str(_predict_id)]
  _str_basename = os.path.basename(filename)
  _str_class_name = _str_basename.strip().split("_")[0]
  _ground_truth_id = int(class_names_to_ids[_str_class_name])

  tf.logging.info('[ylie.Inference.DoIfWrong] ground_truth_id: %d, ground_truth_name: %s, predict_id: %d'% \
    (_ground_truth_id, _str_class_name, _predict_id))

  if _ground_truth_id != _predict_id:
    _dst_dir_file = os.path.join(str_bad_case_dir, _str_wrong_names, _str_basename)
    tf.logging.info('[ylie.Inference.DoIfWrong] Do cp %s to %s'% (filename, os.path.join(str_bad_case_dir, _dst_dir_file)))
    copyfile(filename, _dst_dir_file)
  
def _get_classes_to_labels(dataset_dir):
  labels_file_name = os.path.join(dataset_dir, 'labels.txt')
  class_to_label_dict = {}
  label_to_class_dict = {}
  with open(labels_file_name) as pf:
    for line in pf:
      line_list = line.strip().split(":")
      assert(len(line_list)==2)
      class_to_label_dict[line_list[1]] = line_list[0]
      label_to_class_dict[line_list[0]] = line_list[1]
  return class_to_label_dict, label_to_class_dict

def _get_evalute_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  wd_snapshot_root = os.path.join(dataset_dir, 'wd_snapshot_photo')
  wd_snapshot_evalue = os.path.join(wd_snapshot_root, 'evaluting_dir')
  directories = []
  class_names = []
  for filename in os.listdir(wd_snapshot_root):
    path = os.path.join(wd_snapshot_root, filename)
    if os.path.isdir(path) and filename != "bad_case_dir" and filename != "evaluting_dir":
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for filename in os.listdir(wd_snapshot_evalue):
    path = os.path.join(wd_snapshot_evalue, filename)
    photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


class Inference(object):
  def __init__(self):
    self._inference_image_size = FLAGS.inference_image_size or 299
    self._checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    
  @property
  def image_feed_placeholder(self):
    if hasattr(self, "_image_feed_placeholder"):
       return self._image_feed_placeholder
    self._image_feed_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name="image_feed")

  @property
  def sess(self):
     if hasattr(self, "_sess"):
       return self._sess
     if not hasattr(self, "_image_feed_p"):
       self._image_feed_p = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name="image_feed")
     with slim.arg_scope(inception.inception_v3_arg_scope()):
       self._logits, _ = inception.inception_v3(
            self._image_feed_p, 
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False,
            reuse=False)
       self._predictions = tf.argmax(self._logits, 1)
       saver = tf.train.Saver()
       self._sess = tf.Session()
       saver.restore(self._sess, self._checkpoint_path)
       return self._sess
 
  def _run_predict(self, image_feed): 
     prediction = self.sess.run(self._predictions, feed_dict={self._image_feed_p: image_feed})
     return prediction

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint directory with --checkpoint_path')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    with tf.Session() as sess:
      _inference_obj = Inference()

      #############################
      # Loop the inferencing data #
      #############################
      _snapshot_filenames, _class_names = _get_evalute_filenames_and_classes(FLAGS.dataset_dir)
      _class_names_to_ids, _class_ids_to_names = _get_classes_to_labels(FLAGS.dataset_dir)
     
      _str_bad_case_dir = _check_bad_case_dir(_class_names, FLAGS.dataset_dir)
 
      _image_reader = ImageReader()
  
      #####################################
      # Select the preprocessing function #
      #####################################
      preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          preprocessing_name,
          is_training=False)

      #299 * 299
      inference_image_size = FLAGS.inference_image_size or 299
      
      for _filename in _snapshot_filenames: 
        str_basename = os.path.basename(_filename)
        image_data = tf.gfile.FastGFile(_filename, 'rb').read()
        image = _image_reader.decode_png(image_data)
        image = image_preprocessing_fn(image, inference_image_size, inference_image_size)
        image = tf.expand_dims(image, 0)
        image = sess.run(image)
       
        #tf.logging.info('[ylie.main] class_names: %r, class_names_to_ids: %r' % (_class_names, _class_names_to_ids)) 
        prediction = _inference_obj._run_predict(image)
        tf.logging.info('[ylie.Inference] image_name: %r, prediction: %r' % (str_basename, prediction))
        _action_if_predict_wrong(prediction, _filename, _class_names_to_ids, _class_ids_to_names, _str_bad_case_dir)

if __name__ == '__main__':
  tf.app.run()
