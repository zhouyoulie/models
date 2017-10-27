#!/bin/bash
# Copyright 2017 John Zhou. All Rights Reserved.
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

# Script to preprocess the ai challenge data set.
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See build_mscoco_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./preprocess_aic.sh
set -e

if [ -z "$1" ]; then
  echo "usage preprocess_aic.sh [data dir]"
  exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

## Helper function to download and unpack a .zip file.
#function download_and_unzip() {
#  local BASE_URL=${1}
#  local FILENAME=${2}
#
#  if [ ! -f ${FILENAME} ]; then
#    echo "Downloading ${FILENAME} to $(pwd)"
#    wget -nd -c "${BASE_URL}/${FILENAME}"
#  else
#    echo "Skipping download of ${FILENAME}"
#  fi
#  echo "Unzipping ${FILENAME}"
#  ${UNZIP} ${FILENAME}
#}

cd ${SCRATCH_DIR}

## Download the images.
#BASE_IMAGE_URL="http://msvocds.blob.core.windows.net/coco2014"
#
#TRAIN_IMAGE_FILE="train2014.zip"
#download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
#
#VAL_IMAGE_FILE="val2014.zip"
#download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
TRAIN_IMAGE_DIR="${SCRATCH_DIR}/caption_train_images"
VAL_IMAGE_DIR="${SCRATCH_DIR}/caption_validation_images"
TEST_IMAGE_DIR="${SCRATCH_DIR}/caption_test_images"
#
## Download the captions.
#BASE_CAPTIONS_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
#CAPTIONS_FILE="captions_train-val2014.zip"
#download_and_unzip ${BASE_CAPTIONS_URL} ${CAPTIONS_FILE}
TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/caption_train_annotations.json"
VAL_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/caption_validation_annotations.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_aic_data"

echo "=================debug info by youlie================="
echo "${BUILD_SCRIPT}"
echo "${TRAIN_IMAGE_DIR}"
echo "${VAL_IMAGE_DIR}"
echo "${TRAIN_CAPTIONS_FILE}"
echo "${VAL_CAPTIONS_FILE}"
echo "${OUTPUT_DIR}"
echo "=================debug info by youlie================="

"${BUILD_SCRIPT}" \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_captions_file="${TRAIN_CAPTIONS_FILE}" \
  --val_captions_file="${VAL_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \
