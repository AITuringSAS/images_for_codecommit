"""facade_interface module"""
import os
import sys
import logging


class Inteface(object):
    """FacadeInteface
    This class provide an interface for the automl/efficientdet repository. It set the params definitions
    from params_definitions.py for the command line managment. The automl efficiendet repository is a git submodule 
    that won't be affected by this interface. We propose a Facade Design Pattern to act as mediator between the commands
    and the internal scripts.


    Attributes
    ----------
    pip_tools : PipelineTools
        param definition object from params_definitions.py
    paths : dict
        define the paths to manage the workspace
    args : sys.argv
        parse the command line arguments


    Methods
    -------
    create_tfrecord():
        Creates the tfrecord files from input dataset.

    run_training():
        Starts the training pipeline.

    save_frozen_model():
        Saves the trained model in frozen format.


    Training with command line parameters
    -------
    python3 main.py --URL_DATASET https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip \
    --BATCH_SIZE 1 \
    --BACKBONE_REF efficientdet-d0 \
    --NUM_EPOCHS 5 \
    --MODEL_CKPTS efficientdet-d0-Output-folder


    Training with environment variables    
    -------
    export URL_DATASET=https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip
    export BATCH_SIZE=1
    export BACKBONE_REF=efficientdet-d0
    export NUM_EPOCHS=5
    export MODEL_CKPTS=efficientdet-d0-Output-folder
    python3 main.py 


    Training with configuration file    
    -------
    python3 main.py --configfile params.config

    params.config:
    [Defaults]
    URL_DATASET=https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip
    BACKBONE_REF=efficientdet-d0
    BATCH_SIZE=1
    NUM_EPOCHS=5
    MODEL_CKPTS=efficientdet-d0-output
    """

    def __init__(self, args_tfrecords, args_efficientdet, args_freeze, paths, files):
        logging.info("=================================")
        logging.info("--> Setting params definitions...please...wait!")
        self.args_tfr = args_tfrecords
        self.args_efd = args_efficientdet
        self.args_fre = args_freeze
        self.paths = paths
        self.files = files

    def exit_code_to_bool(self, exit_code):
        if exit_code == 0:
            return True
        else:
            return False

    def create_tfrecord(self):
        """create_tfrecord method
        This method define the command line flags in the main script for the creation of the TFRecords at
        repository/tfRecordMod/create_tfrecords.py

        Parameters:
            None

        Returns:
            None
        """
        print("\n*************************************")
        print("*** TFRECORD SCRIPT *** ")
        print("*************************************")
        print()

        COMMAND = """{} {} \
        --data_dir={} \
        --output_path={} \
        --num_shards={} \
        --path_metadata={} \
        --autoaugment_policy={} \
        --train_valid_split={}
        """.format(sys.executable, self.files['TFRECORD_SCRIPT'], self.args_tfr.data_dir, self.args_tfr.output_path, self.args_tfr.num_shards, self.args_tfr.path_metadata, self.args_tfr.autoaugment_policy, self.args_tfr.train_valid_split)

        #-------------------- Start TFrecord conversion
        exit_code = os.system(COMMAND)
        print()
        return self.exit_code_to_bool(exit_code)

    def run_training(self):
        """run_training method
        This method define the command line flags in the main script for the training step at
        repository/automl/efficientdet/main.py

        Parameters:
            None

        Returns:
            None
        """
        print("\n*************************************")
        print("*** TRAINING SCRIPT *** ")
        print("*************************************")
        print("--> Training...please...wait!")
        print()

        # Commands that you must set/override by code and not by command line
        # strategy = [tpu/gpus]
        # gpc_project

        # =================================
        # mode: train
        # =================================
        if self.args_efd.mode == 'train':
            COMMAND = """{} {} \
            --mode={} \
            --train_file_pattern={} \
            --model_dir={} \
            --pretrained_ckpt={} \
            --batch_size={} \
            --num_epochs={} \
            --hparams={} \
            --num_examples_per_epoch={} \
            --profile={} \
            --tpu={} \
            --tpu_zone={} \
            --use_xla={} \
            --num_cores={}
            """.format(sys.executable, self.files['EFFICIENTDET_MAIN_SCRIPT'],
                       #-------------------- Training standard parameters
                       self.args_efd.mode, self.args_efd.train_file_pattern,
                       self.args_efd.model_dir, self.args_efd.pretrained_ckpt, self.args_efd.batch_size, self.args_efd.num_epochs,
                       self.args_efd.hparams, self.args_efd.num_examples_per_epoch, self.args_efd.profile,
                       #-------------------- TPU support
                       self.args_efd.tpu, self.args_efd.tpu_zone,
                       self.args_efd.use_xla, self.args_efd.num_cores,
                       #-------------------- Model evaluation after train (optional)
                       )
        # =================================
        # mode: traineval
        # =================================
        else:
            COMMAND = """{} {} \
            --mode={} \
            --train_file_pattern={} \
            --model_dir={} \
            --pretrained_ckpt={} \
            --batch_size={} \
            --num_epochs={} \
            --hparams={} \
            --num_examples_per_epoch={} \
            --profile={} \
            --tpu={} \
            --tpu_zone={} \
            --use_xla={} \
            --num_cores={} \
            --val_file_pattern={} \
            --eval_samples={}
            """.format(sys.executable, self.files['EFFICIENTDET_MAIN_SCRIPT'],
                       #-------------------- Training standard parameters
                       self.args_efd.mode, self.args_efd.train_file_pattern,
                       self.args_efd.model_dir, self.args_efd.pretrained_ckpt, self.args_efd.batch_size, self.args_efd.num_epochs,
                       self.args_efd.hparams, self.args_efd.num_examples_per_epoch, self.args_efd.profile,
                       #-------------------- TPU support
                       self.args_efd.tpu, self.args_efd.tpu_zone,
                       self.args_efd.use_xla, self.args_efd.num_cores,
                       #-------------------- Model evaluation support
                       self.args_efd.val_file_pattern, self.args_efd.eval_samples
                       )

        #-------------------- Start training
        exit_code = os.system(COMMAND)

        print()
        print("--> training has finished!")
        print()

        return self.exit_code_to_bool(exit_code)

    def save_frozen_model(self):
        """save_frozen method
        This method define the command line flags in the main script to export the frozen model at
        repository/freezeModelMod/freeze_aituring.py

        Parameters:
            None

        Returns:
            None
        """
        print("\n*************************************")
        print("*** FREEZE MODEL SCRIPT *** ")
        print("*************************************")
        print("--> Saving checkpoints as frozen_model...please...wait!")

        COMMAND = """{} {} \
        --path_ckpt={} \
        --path_yaml={} \
        --path_output={} \
        --model_name_={}
        """.format(sys.executable, self.files['FREEZE_MAIN_SCRIPT'],
                   self.args_fre.path_ckpt,
                   self.args_fre.path_yaml,
                   self.args_fre.path_output,
                   self.args_fre.model_name_)

        #-------------------- Start freezing model
        exit_code = os.system(COMMAND)

        print("--> frozen model saved!")
        print()
        return self.exit_code_to_bool(exit_code)
