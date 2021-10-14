"""
main script to create tfrecords

It is recommended that each shard should be 100–200mb.
reference: https://medium.com/@rodrigobrechard/tfrecords-how-to-use-sharding-94059e2b2c6b
"""
import hashlib
import io
import os
import sys
import glob
import random

from absl import app
from absl import flags
from absl import logging
from lxml import etree

import PIL.Image
import PIL.ImageOps
import tensorflow.compat.v1 as tf
import cv2
import json

# custom modules
import tfrecord_util
import create_yaml

# remove FutureWarning: The  behavior of  this  method  will change in  future
# versions. Use specific 'len(elem)' or 'elem is not None' test instead.
import warnings
warnings.filterwarnings("ignore")


flags.DEFINE_string('data_dir', 'temp', 'Root directory to raw dataset')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('path_metadata', os.getcwd(), 'Path to save metadata info')
flags.DEFINE_integer('num_shards', -1, 'Number of shards for output file.')
flags.DEFINE_string('autoaugment_policy', '', 'Define the type of data augmentation')
flags.DEFINE_float('train_valid_split', 1, 'Define the percentage to divide the training and validation set')

FLAGS = flags.FLAGS

# global image id
GLOBAL_IMG_ID = 0
# global annotation id
GLOBAL_ANN_ID = 0


def unique_list(list_):
    """unique_list

    Args:
        list_: input_list

    Returns:
        list
    """
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list_:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def splitall(path):
    """splitall

    common function from automl/efficientdet/create_coco_tfrecords.py

    Args:
        path: str

    Returns:
        list
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        # sentinel for absolute paths
        if parts[0] == path:
            allparts.insert(0, parts[0])
            break
        # sentinel for relative paths
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def create_labelmapDict_patch(list_all_images, path_dataset):
    """create_labelmapDict_patch

    This function maps an input dataset to labelmap

    Args:
        list_all_images: input list
        path_dataset: input dir

    Returns:
        labelmap_: dict
    """
    list_all_classes = []
    for idx, name_image_ in enumerate(list_all_images):
        _, tail = os.path.split(name_image_)
        temp_obj = []
        name_file_xml_all = os.path.join(path_dataset, 'LABELS', tail[0:-3] + 'xml')
        if os.path.exists(name_file_xml_all):
            with tf.gfile.GFile(name_file_xml_all, 'rb') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
            if 'object' in data:
                for obj in data['object']:
                    name_in_obj_ = obj['name'].replace(' ', '').strip()
                    if name_in_obj_ != 'INCOMPLETAS':
                        list_all_classes.append(name_in_obj_)
                        temp_obj.append(obj)
    # list_all_classes = unique_list(list_all_classes)
    list_all_classes = list(set(list_all_classes))
    list_all_classes.sort()
    list_all_classes.insert(0, 'background')
    labelmap_ = {el: k for k, el in enumerate(list_all_classes)}
    return labelmap_


def get_image_id(filename):
    """get_image_id

    function to get image by id

    Args:
        filename: input filename

    Returns:
        global
    """
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def dict_to_tf_example(data,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """dict_to_tf_Example

    This function convert an input dictionary to tf.example

    Args:
        data: [description]
        label_map_dict: [description]
        ignore_difficult_instances: [description] (default: {False})

    Returns:
        None
    """
    full_path = os.path.join(FLAGS.data_dir, 'IMAGENES', data['filename'])[0:-3] + 'jpg'
    image_ = cv2.imread(full_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    image_id = get_image_id(data['filename'])
    width = int(image_.shape[1])
    height = int(image_.shape[0])
    image_id = get_image_id(data['filename'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    area = []
    classes = []
    classes_text = []
    if 'object' in data:
        for obj in data['object']:
            name_in_obj_ = obj['name'].replace(' ', '').strip()
            if name_in_obj_ in label_map_dict:
                x_pos = [int(obj['bndbox']['xmax']), int(obj['bndbox']['xmin'])]
                y_pos = [int(obj['bndbox']['ymax']), int(obj['bndbox']['ymin'])]
                xmin.append((float(min(x_pos))) / width)
                ymin.append((float(min(y_pos))) / height)
                xmax.append((float(max(x_pos))) / width)
                ymax.append((float(max(y_pos))) / height)
                area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
                classes_text.append(name_in_obj_.replace(' ', '').encode('utf8'))
                classes.append(int(label_map_dict[name_in_obj_]))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                tfrecord_util.int64_feature(height),
                'image/width':
                tfrecord_util.int64_feature(width),
                'image/filename':
                tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                tfrecord_util.bytes_feature(encoded_jpg),
                'image/format':
                tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                tfrecord_util.float_list_feature(ymax),
                'image/object/area':
                tfrecord_util.float_list_feature(area),
                'image/object/class/text':
                tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                tfrecord_util.int64_list_feature(classes),
            }))
    return example


def save_to_tfrecord(type_dataset, dataset, label_map_dict, labels_by_class_dict):
    """save_to_tfrecord function

    This function save a set of images in a shard

    Args:
        type_dataset: train or eval
        dataset: list of images
        label_map_dict: labelmap 
        labels_by_class_dict: classes from labelmap
    """

    # create empty shards
    writers = [
        tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, type_dataset) + '-%05d-of-%05d.tfrecord' %
                                    (i, FLAGS.num_shards))
        for i in range(FLAGS.num_shards)
    ]

    # add images to empty shards
    for idx, name_image_ in enumerate(dataset):
        _, tail = os.path.split(name_image_)
        data_total = {}
        name_file_xml_all = os.path.join(FLAGS.data_dir, 'LABELS', tail[0:-3] + 'xml')
        if os.path.exists(name_file_xml_all):
            with tf.gfile.GFile(name_file_xml_all, 'rb') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data_total = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
            if 'object' in data_total:
                for obj in data_total['object']:
                    name_in_obj_ = obj['name'].replace(' ', '')
                    if name_in_obj_ in labels_by_class_dict:
                        labels_by_class_dict[name_in_obj_] += 1
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, int(len(dataset)))
        if bool(data_total):
            tf_example = dict_to_tf_example(
                data_total,
                label_map_dict)
            writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

    # close opened shards
    for writer in writers:
        writer.close()


def main(_):
    """
    main function

    The following steps are:
    1. read images from input directory
    2. suffle list of images
    3. calculate number of shards for list of images
    4. split dataset in train and test
    5. create labelmap
    6. save train images in tfrecord
    7. save test images in tfrecord
    8. save metadata information
    """

    # files path
    files = {
        'IMAGES': os.path.join(FLAGS.data_dir, 'IMAGENES', '*.jpg'),
        'HPARAMS_CONFIG': os.path.join(FLAGS.output_path, 'hparams_config.yaml'),
        'TEST_IMAGES': os.path.join(FLAGS.output_path, 'list_img_test.txt'),
        'METADATA': os.path.join(FLAGS.path_metadata, 'metadata.json')
    }

    # list of images
    list_all_images = glob.glob(files['IMAGES'])
    if not list_all_images:
        print("--> not found images in {} or format is not the same as IMAGENES-LABELS".format(FLAGS.data_dir))
        sys.exit(-1)

    # shuffle list of images
    random.shuffle(list_all_images)

    # calculate shards, each 800 images
    n_images_shard = 800
    n_shards = int(len(list_all_images) / n_images_shard) + (1 if len(list_all_images) % 800 != 0 else 0)
    if FLAGS.num_shards < 1:
        FLAGS.num_shards = n_shards
        n_images_shard = int(len(list_all_images) / n_shards)

    num_samp_train = int(FLAGS.train_valid_split * len(list_all_images))

    # split dataset into train, validation and test
    list_img_train = list_all_images[0:num_samp_train]
    list_img_remaining = list_all_images[num_samp_train::]

    # split remaining list to validation-test by 50%
    num_samp_val = int(0.5 * len(list_img_remaining))
    list_img_val = list_img_remaining[0:num_samp_val]
    list_img_test = list_img_remaining[num_samp_val::]

    # create label_map
    label_map_dict = create_labelmapDict_patch(list_all_images, FLAGS.data_dir)
    class_main_names = list(label_map_dict.keys())[1::]
    labels_by_class_dict = dict.fromkeys(class_main_names, 0)
    create_yaml.create_yaml(class_main_names, files['HPARAMS_CONFIG'],
                            FLAGS.autoaugment_policy, sample_image_path=list_img_test[0:2])

    # save images in train tfrecord
    print('--> saving train tfrecords...')
    save_to_tfrecord(type_dataset='train', dataset=list_img_train,
                     label_map_dict=label_map_dict, labels_by_class_dict=labels_by_class_dict)
    print()

    # save images in val tfrecord
    if list_img_val:
        print('--> saving validation tfrecords...')
        save_to_tfrecord(type_dataset='eval', dataset=list_img_val,
                         label_map_dict=label_map_dict, labels_by_class_dict=labels_by_class_dict)
        print()

    # save images list for testing
    if list_img_test:
        with open(files['TEST_IMAGES'], 'w') as out_txt:
            out_txt.writelines("%s\n" % img for img in list_img_test)

    # save labelmap to metadata file
    with open(files['METADATA'], 'r+') as file:
        file_data = json.load(file)
        file_data.update(
            {
                'label_map': label_map_dict,
                'number_of_images': {
                    'total': len(list_all_images),
                    'train': len(list_img_train),
                    'val:': len(list_img_val),
                    'test': len(list_img_test)
                },
                'train_valid_split': FLAGS.train_valid_split,
                'valid_test_split': 0.5,
                'autoaugment_policy': FLAGS.autoaugment_policy,
                'number_of_shards': FLAGS.num_shards,
                'number_images_per_shard': 800
            })
        file.seek(0)
        json.dump(file_data, file, indent=4)

    print('~~ SUMMARY ~~')
    print('--> Images:                  {}'.format(len(list_all_images)))
    print('--> Images for training:     {}'.format(len(list_img_train)))
    print('--> Images for validation:   {}'.format(len(list_img_val)))
    print('--> Images for testing:      {}'.format(len(list_img_test)))


if __name__ == '__main__':
    app.run(main)
