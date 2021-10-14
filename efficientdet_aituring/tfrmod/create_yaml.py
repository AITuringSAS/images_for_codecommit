"""
create_yaml module

This module provide a create_yaml function to interface the efficientdet-configuration file
"""

import yaml


def create_yaml(list_classes, path_output, autoaugment_policy, sample_image_path=None):
  """create_yaml function

  This function has the efficientdet hyper-parameters

  Args:
    list_classes: list
    path_output: str
    autoaugment_policy: str
  """
  dict_classes = {(id_ + 1): name_class_ for id_,
                  name_class_ in enumerate(list_classes)}
  dict_file = {
      # dataset specific parameters
      'num_classes': len(list_classes) + 1,
      'label_map': dict_classes,
      'max_instances_per_image': 300,

      # input preprocessing parameters
      'autoaugment_policy': autoaugment_policy,
      'map_freq': 1,
      'sample_image': sample_image_path,

      # optimization
      'learning_rate': 0.004,
      'lr_warmup_init': 0.0004,

      # localization loss
      'iou_loss_type': 'ciou',

      # No stochastic depth in default.
      'moving_average_decay': 0,
      'img_summary_steps': 5,

      # For post-processing nms, must be a dict.
      'nms_configs': {
          'method': 'gaussian',
          'iou_thresh': 0.6,
          'score_thresh': 0.02,
          'sigma': .3,
          'max_nms_inputs': 0,
          # maximum number of predictions per image
          'max_output_size': 100
      },

      # A temporary flag to switch between legacy and keras models.
      'use_keras_model': True,

      # Parameters for the Checkpoint Callback.
      'save_freq': 'epoch',
      'verbose': 0,
  }
  with open(path_output, 'w') as file:
    _ = yaml.dump(dict_file, file)
