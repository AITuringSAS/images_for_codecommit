"""
main script to interface automl/efficientdet
"""
import os
import sys
import json
import shutil
from datetime import datetime
from os.path import expanduser

# Append submodule path to python search path
sys.path.append(os.path.join(os.getcwd(), 'freeze'))
sys.path.append(os.path.join(os.getcwd(), 'tfrmod'))
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientdet'))

# Tools to parse commands, zip and tar.gz file
import tarfile
from zipfile import ZipFile
from absl.flags import argparse_flags

# Custom submodules
from pipeline import command_interface
from automl.efficientdet.tf2.train import define_flags as define_flags_train_script_tf2
from automl.efficientdet.tf2.train import FLAGS as flags_efficientdet
from tfrmod.create_tfrecords import FLAGS as flags_tfrecord
from freeze.freeze_model import FLAGS as flags_freeze


def download_and_uncompress_backbone(backbone_name, backbone_url, backbone_save_dir):
    """
    define_parameters method
        This method download a pre-trained model 

        Parameters:
            backbone_name (str): model name
            backbone_url (str): backbone url
            backbone_save_dir (str): path to save backbone

        Returns:
            None
    """
    # Download backbone checkpoints
    os.system(' '.join(['wget', '--no-check-certificate', '-q', '--show-progress', '--progress=bar:force', backbone_url, '-O',
                        os.path.join(backbone_save_dir, backbone_name + 'tar.gz')]))

    # Uncompress backbone .tar.gz files
    tar = tarfile.open(os.path.join(backbone_save_dir, backbone_name + 'tar.gz'))
    tar.extractall(backbone_save_dir)
    tar.close()

    # Delete .tar.gz backbone file
    os.remove(os.path.join(backbone_save_dir, backbone_name + 'tar.gz'))


def download_and_uncompress_dataset(dataset_url, save_dir, dataset_name):
    """
    Function to download and uncompress dataset from S3

    Parameters:
            dataset_url (str): url from S3
            save_dir (str): path to download dataset
            dataset_name (str): dataset filename to save

        Returns:
            None
    """

    # Download dataset from S3 (dataset must be public)
    os.system(' '.join(['wget', '--no-check-certificate', '-q', '--show-progress',
                        '--progress=bar:force', dataset_url, '-O', dataset_name]))
    # Unzip dataset
    with ZipFile(dataset_name, 'r') as zipobj:
        zipobj.extractall(save_dir)
    # Delete .zip dataset
    os.remove(dataset_name)


def main():
    """Pipeline

    The following pipeline is the standard procedure 

    1. Set params definitions
    2. Set paths to workspace
    3. Download dataset
    4. Download backbone checkpoint
    5. run create_tfrecord method
    6. run training method
    7. save frozen model 
    """
    # get the flags definitions from automl.efficientdet.tf2.train
    define_flags_train_script_tf2()

    # datetime object containing current date and time
    now = datetime.now()

    # =================================
    # Add default flags
    # =================================
    parser = argparse_flags.ArgumentParser(
        description='Command Line Tools EfficientDet Interface.', fromfile_prefix_chars='@')

    # check if not arguments were given
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    parser.add_argument('--url_dataset', help='url e.g. S3, GCS, etc.', metavar='')
    parser.add_argument('--backbone_ref', help='backbone name e.g. efficientdet-d0', metavar='')
    parser.add_argument(
        '--model_ckpts', help='(optional) path to save train model. (default: saved in workspace/result)', metavar='', default=None)

    # Dictionary to set workspace
    home = expanduser("~")
    workspace = {'WORKSPACE': os.path.join(home, 'workspace')}

    # Dictionary for managing paths
    paths = {
        'DATASET_DIR': os.path.join(workspace['WORKSPACE'], 'dataset'),
        'MODEL_OUTPUT_DIR': os.path.join(workspace['WORKSPACE'], 'result'),
        'FREEZE_MODEL_DIR': os.path.join(workspace['WORKSPACE'], 'freeze_model'),
        'SAVED_MODEL_DIR': os.path.join(workspace['WORKSPACE'], 'freeze_model', 'saved_model'),
        'TFRECORD_DIR': os.path.join(workspace['WORKSPACE'], 'tfrecords'),
        'BACKBONE_CKPT_DIR': os.path.join(workspace['WORKSPACE'], 'ckpt'),
        'METADATA_DIR': workspace['WORKSPACE']
    }

    # Dictionary for managing files
    files = {
        'TFRECORD_SCRIPT': os.path.join(os.getcwd(), 'tfrmod', 'create_tfrecords.py'),
        'EFFICIENTDET_MAIN_SCRIPT': os.path.join(os.getcwd(), 'automl', 'efficientdet', 'tf2', 'train.py'),
        'FREEZE_MAIN_SCRIPT': os.path.join(os.getcwd(), 'freeze', 'freeze_model.py'),
        'TFRECORD_TRAIN_FILES': os.path.join(paths['TFRECORD_DIR'], 'train*.tfrecord'),
        'TFRECORD_TEST_FILES': os.path.join(paths['TFRECORD_DIR'], 'eval*.tfrecord'),
        'HPARAMS_YAML': os.path.join(paths['TFRECORD_DIR'], 'hparams_config.yaml'),
        'DATASET_FILE': os.path.join(paths['DATASET_DIR'], 'dataset.zip'),
    }

    # Dictionary for backbone url managment
    # reference
    # https://github.com/google/automl/tree/master/efficientdet#2-pretrained-efficientdet-checkpoints
    backbone_url = {
        'efficientdet-d0': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz',
        'efficientdet-d1': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.tar.gz',
        'efficientdet-d2': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.tar.gz',
        'efficientdet-d3': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.tar.gz',
        'efficientdet-d4': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz',
        'efficientdet-d5': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d5.tar.gz',
        'efficientdet-d6': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d6.tar.gz',
        'efficientdet-d7': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7.tar.gz',
        'efficientdet-d7x': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7x.tar.gz'
    }

    # =================================
    # Setting default flags to workspace
    # =================================
    # TFRECORD FLAGS
    flags_tfrecord.data_dir = paths['DATASET_DIR']
    flags_tfrecord.output_path = paths['TFRECORD_DIR']
    flags_tfrecord.path_metadata = paths['METADATA_DIR']

    # EFFICIENTDET FLAGS
    flags_efficientdet.mode = 'train'
    flags_efficientdet.train_file_pattern = files['TFRECORD_TRAIN_FILES']
    flags_efficientdet.val_file_pattern = files['TFRECORD_TEST_FILES']
    flags_efficientdet.model_dir = paths['MODEL_OUTPUT_DIR']
    flags_efficientdet.pretrained_ckpt = os.path.join(paths['BACKBONE_CKPT_DIR'], 'efficientdet-d1')
    flags_efficientdet.hparams = files['HPARAMS_YAML']

    # FREEZE MODEL FLAGS
    flags_freeze.path_ckpt = paths['MODEL_OUTPUT_DIR']
    flags_freeze.path_yaml = files['HPARAMS_YAML']
    flags_freeze.path_output = paths['FREEZE_MODEL_DIR']

    # =================================
    # Default flags will be overridden if provided by command line
    # =================================
    args = parser.parse_args()

    # =================================
    # catch environment flags if not detected by command line or config.file
    # =================================
    env_flags = [os.getenv('url_dataset'), os.getenv('backbone_ref'), os.getenv('num_epochs'), os.getenv(
        'batch_size'), os.getenv('model_ckpts'), os.getenv('num_examples_per_epoch')]

    if all(env_flags):
        args.url_dataset = env_flags[0]
        args.backbone_ref = env_flags[1]
        flags_efficientdet.num_epochs = int(env_flags[2])
        flags_efficientdet.batch_size = int(env_flags[3])
        args.model_ckpts = env_flags[4]
        flags_efficientdet.num_examples_per_epoch = int(env_flags[5])

    # =================================
    # Create workspace directories
    # =================================
    # clean workspace
    try:
        shutil.rmtree(workspace['WORKSPACE'])
    except OSError:
        # print("Error: %s - %s." % (e.filename, e.strerror))
        pass

    # add folders to workspace
    try:
        os.mkdir(workspace['WORKSPACE'])
        os.mkdir(paths['DATASET_DIR'])
        os.mkdir(paths['MODEL_OUTPUT_DIR'])
        os.mkdir(paths['FREEZE_MODEL_DIR'])
        os.mkdir(paths['SAVED_MODEL_DIR'])
        os.mkdir(paths['TFRECORD_DIR'])
        os.mkdir(paths['BACKBONE_CKPT_DIR'])
    except OSError:
        pass

    # =================================
    # Update default flags
    # =================================
    # EFFICIENTDET FLAGS
    flags_efficientdet.model_name = args.backbone_ref
    flags_efficientdet.pretrained_ckpt = os.path.join(paths['BACKBONE_CKPT_DIR'], args.backbone_ref)
    if args.model_ckpts:
        flags_efficientdet.model_dir = os.path.join(args.model_ckpts, 'model')

    # FREEZE MODEL FLAGS
    flags_freeze.model_name_ = flags_efficientdet.model_name
    flags_freeze.path_ckpt = flags_efficientdet.model_dir
    if args.model_ckpts:
        flags_freeze.path_output = os.path.join(args.model_ckpts, 'freeze_model')

    # Download dataset from S3
    print('\n--> downloading dataset...')
    download_and_uncompress_dataset(args.url_dataset, paths['DATASET_DIR'], files['DATASET_FILE'])

    # Download backbone checkpoints
    print()
    print('--> downloading backbone checkpoint {}'.format(args.backbone_ref))
    download_and_uncompress_backbone(args.backbone_ref, backbone_url[args.backbone_ref], paths['BACKBONE_CKPT_DIR'])

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    with open(os.path.join(paths['METADATA_DIR'], 'metadata.json'), 'w') as outfile:
        final_json = {
            'url': args.url_dataset,
            'backbone_reference': args.backbone_ref,
            'epochs': int(flags_efficientdet.num_epochs),
            'train_batch_size': int(flags_efficientdet.batch_size),
            'num_examples_per_epoch': int(flags_efficientdet.num_examples_per_epoch),
            'date': dt_string,
            'mode': flags_efficientdet.mode
        }
        json.dump(final_json, outfile, indent=4)
    print()

    # Command Interface tf2
    commandI = command_interface.Inteface(args_tfrecords=flags_tfrecord,
                                          args_efficientdet=flags_efficientdet,
                                          args_freeze=flags_freeze,
                                          paths=paths,
                                          files=files)

    # 1. Create tfrecord files
    success = commandI.create_tfrecord()
    if not success:
        sys.exit(-1)

    # 2. Start training pipeline
    success = commandI.run_training()
    if not success:
        sys.exit(-1)

    # 3. Save trained model
    success = commandI.save_frozen_model()
    if not success:
        sys.exit(-1)


if __name__ == '__main__':
    print("\n*************************************")
    print("*** MAIN SCRIPT *** ")
    print("*************************************")
    main()

    print()
    for i in range(5):
        print('*' * i)
    print()
    print("until next time baby ;)")
    print()
    for i in range(5, 0, -1):
        print('*' * i)
