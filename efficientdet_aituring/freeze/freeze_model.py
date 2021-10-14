"""
main script to export model to savedModel and frozenModel
"""
import os
import sys
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '..', 'automl', 'efficientdet'))
import inference
import yaml
import tensorflow.compat.v1 as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
from absl import app
from absl import flags
from absl import logging


flags.DEFINE_string(
    'path_ckpt',
    default=None,
    help='Path to the trained checkpoint')
flags.DEFINE_string(
    'path_yaml',
    default=None,
    help='Path to the created yaml file, this file is created during training')
flags.DEFINE_string(
    'path_output',
    default=None,
    help='Path for saving the freeze model.')
flags.DEFINE_string(
    'model_name_',
    default='efficientdet-d2',
    help='Name of model to freeze.')

FLAGS = flags.FLAGS


def parse_from_yaml(yaml_file_path):
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


def main(_):
    """
    main function

    The following steps are:
    1. read checkpoint from input directory
    2. read hparams_config.yaml from input directory
    3. load model from checkpoints
    4. export model to save_model.pb format
    5. export model to frozen_model.pb format
    """
    config = parse_from_yaml(FLAGS.path_yaml)
    driver = inference.ServingDriver(FLAGS.model_name_, FLAGS.path_ckpt, batch_size=1, model_params=config)
    driver.build()
    driver.export(FLAGS.path_output)

    # save hparams_config.h to freeze_model folder
    tf.io.gfile.copy(FLAGS.path_yaml, os.path.join(FLAGS.path_output, 'hparams_config.yaml'), overwrite=True)


if __name__ == '__main__':
    logging.set_verbosity(logging.ERROR)
    app.run(main)
