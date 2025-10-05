#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio DistilBERT Upstream Training Entry Point
==============================================

Main script for upstream (teacher-student) training in the Audio DistilBERT framework.
This module orchestrates the training pipeline for distilling knowledge from a teacher
BERT-style model to a smaller student model using masked acoustic modeling.

Key Features:
- Teacher-student distillation training
- Masked acoustic modeling (MAM) support
- Multi-GPU training capability
- Configurable model architectures via YAML
- Comprehensive logging and checkpointing

Author: fanfan-yu
Date: 2025.10.05

Usage:
    python main_upstream.py --student_config config/distiller_upstream_fbankBase.yaml \
                           --teacher_resume path/to/teacher/checkpoint.ckpt
"""

import os
import glob
import yaml
import torch
import random
import argparse
import numpy as np
from shutil import copyfile
from argparse import Namespace
from dataloader.dataloader import get_Dataloader
from upstream.utils import parse_prune_heads

import logging
import time

# CUDA device configuration for multi-GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 6, 7"

#################
# PATH HANDLING #
#################
import sys

S3PRL_PATH = os.getcwd()  # or set this to your own path that points to the S3PRL repo
if S3PRL_PATH not in sys.path:
    sys.path.append(S3PRL_PATH)


######################
# UPSTREAM ARGUMENTS #
######################
def get_upstream_args():
    """
    Parse command-line arguments for upstream training configuration.
    
    This function defines all configurable parameters for the Audio DistilBERT
    training process, including model paths, training settings, and dataset options.
    
    Returns:
        tuple: (args, teacher_config, student_config) where
            - args: Parsed command-line arguments
            - teacher_config: Teacher model configuration dict from YAML
            - student_config: Student model configuration dict from YAML
    
    Configuration Sources:
    - Student model: Loaded from --student_config YAML file
    - Teacher model: Loaded from --teacher_resume checkpoint or --teacher_config YAML
    """
    parser = argparse.ArgumentParser(
        description='Audio DistilBERT: Teacher-Student Distillation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python main_upstream.py --student_config config/distiller.yaml --teacher_resume teacher.ckpt
  
  # Multi-GPU training
  python main_upstream.py --multi_gpu --student_config config/distiller.yaml --teacher_resume teacher.ckpt
  
  # Resume training
  python main_upstream.py --student_resume student.ckpt --teacher_resume teacher.ckpt
        """)

    parser.add_argument('--run', default='transformer', choices=['transformer'], help='Select pre-training task. \
                        For the transformer models, which type of pre-training (mockingjay, tera, aalbert, etc) \
                        is determined by config file.')
    # Model configuration paths
    parser.add_argument('--student_config', 
                        default='config/distiller_upstream_fbankBase.yaml', 
                        type=str,
                        help='Path to student model configuration YAML file')
    parser.add_argument('--teacher_resume', 
                        default='result/result_transformer/fbank_test_six_transformer/states-200000.ckpt', 
                        help='Path to teacher model checkpoint for distillation')

    # Training configuration
    parser.add_argument('--name', 
                        default='mockingjay_fbankBase', 
                        type=str, 
                        help='Experiment name for logging and checkpointing')
    parser.add_argument('--ckpdir', 
                        default='result/result_transformer/distiller_six_transformer_one/', 
                        type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('--seed', 
                        default=1337, 
                        type=int, 
                        help='Random seed for reproducible training')

    # Runtime options
    parser.add_argument('--test', 
                        default='', 
                        type=str, 
                        help='Path to saved model for testing/inference')
    parser.add_argument('--cpu', 
                        action='store_true', 
                        help='Force CPU-only training (disable GPU)')
    parser.add_argument('--multi_gpu', 
                        default=True, 
                        action='store_true', 
                        help='Enable multi-GPU training via DataParallel')
    parser.add_argument('--test_reconstruct', 
                        action='store_true', 
                        help='Test reconstruction capability on unmasked data')

    # Distillation hyperparameters
    parser.add_argument('--temperature', 
                        default=2.0, 
                        type=float,
                        help='Temperature for knowledge distillation softmax')
    parser.add_argument('--parent', 
                        action='store_true', 
                        help='Train the parent/teacher model instead of student')
    parser.add_argument('--student_resume', 
                        default=None,
                        help='Path to student checkpoint to resume training')

    # parse
    args = parser.parse_args()

    # student train
    setattr(args, 'gpu', not args.cpu)
    student_config = yaml.load(open(args.student_config, 'r'), Loader=yaml.FullLoader)
    parse_prune_heads(student_config)

    # teacher load
    if args.parent:
        assert args.run is not None and args.teacher_config is not None, '`--run` and `--config` must be given if `--resume` is not provided'
        setattr(args, 'gpu', not args.cpu)
        teacher_config = yaml.load(open(args.teacher_config, 'r'), Loader=yaml.FullLoader)
        parse_prune_heads(teacher_config)
    else:
        if os.path.isdir(args.teacher_resume):
            ckpts = glob.glob(f'{args.teacher_resume}/*.ckpt')
            assert len(ckpts) > 0
            ckpts = sorted(ckpts, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            resume_ckpt = ckpts[-1]
        else:
            resume_ckpt = args.teacher_resume

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        ckpt = torch.load(resume_ckpt, map_location='cpu')
        # args = update_args(args, ckpt['Settings']['Paras'])
        teacher_config = ckpt['Settings']['Config']
        setattr(args, 'resume', resume_ckpt)

    return args, teacher_config, student_config


##################
# GET DATALOADER #
##################
def get_dataloader(args, config):
    if not os.path.exists(config['dataloader']['data_path']):
        raise RuntimeError('[run_upstream] - Data path not valid:', config['dataloader']['data_path'])
    print('[run_upstream] - Loading input data: ' + str(config['dataloader']['train_set']) + ' from ' +
          config['dataloader']['data_path'])
    print('[run_upstream] - getting train dataloader...')

    # select mode
    dataloader = get_Dataloader(split='train', load='acoustic', use_gpu=args.gpu,
                                mam_config=config['transformer'], **config['dataloader'], **config)

    return dataloader


###################
# RUN TRANSFORMER #
###################
def run_transformer(args, teacher_config, student_config, logger):
    """
    Execute the complete teacher-student distillation training pipeline.
    
    This function orchestrates the entire training workflow including:
    1. Directory setup for checkpoints and logging
    2. Model initialization (teacher and student)
    3. Data loading with masked acoustic modeling
    4. Knowledge distillation training loop
    
    Args:
        args (Namespace): Command-line arguments
        teacher_config (dict): Teacher model configuration
        student_config (dict): Student model configuration  
        logger (Logger): Configured logging instance
    
    Checkpoint Management:
    - Creates experiment directory if it doesn't exist
    - Copies configuration files for reproducibility
    - Saves latest checkpoint regularly during training
    """
    from upstream.runner import Runner

    # Setup checkpoint directory
    if args.ckpdir == '':
        if args.name is None: 
            args.name = 'run_' + str(random.randint(0, 999))
        ckpdir = os.path.join('result/result_transformer/', args.name)
    else:
        ckpdir = args.ckpdir
        
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    
    # Copy configuration files for reproducibility
    copyfile(args.student_config, 
             os.path.join(ckpdir, args.student_config.split('/')[-1]))
    copyfile(args.teacher_resume, 
             os.path.join(ckpdir, args.teacher_resume.split('/')[-1]))

    # Initialize data pipeline
    dataloader = get_dataloader(args, student_config)

    # Initialize training runner
    runner = Runner(args, teacher_config, student_config, dataloader, ckpdir, logger)
    runner.set_model()
    runner.train()


########
# MAIN #
########
def main():
    """
    Main entry point for Audio DistilBERT upstream training.
    
    This function initializes the complete training pipeline with proper
    random seeding, logging configuration, and error handling.
    
    Pipeline:
    1. Parse command-line arguments and configurations
    2. Set reproducible random seeds
    3. Configure comprehensive logging
    4. Execute training process
    """
    # Parse configuration
    args, teacher_config, student_config = get_upstream_args()

    # Set reproducible random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Configure comprehensive logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = args.ckpdir + '/log_fbank_' + args.run + '.txt'
    
    # File handler for persistent logs
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)
    
    # Console handler for real-time monitoring
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log initialization information
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("Training arguments: %s", str(args))
    logger.info("\nTeacher configuration: %s", str(teacher_config))
    logger.info("\nStudent configuration: %s", str(student_config))

    # Execute training pipeline
    print("[Main] Starting Audio DistilBERT upstream training...")
    run_transformer(args, teacher_config, student_config, logger)
    print("[Main] Training completed!")


if __name__ == '__main__':
    main()