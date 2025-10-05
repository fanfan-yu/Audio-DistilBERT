# -*- coding: utf-8 -*-
"""
Audio-DistilBERT Data Loader Module
===================================

This module provides comprehensive data loading utilities for the Audio DistilBERT project,
a distilled speech representation learning framework. It handles various datasets including
LibriSpeech for pre-training and downstream tasks.

Key Features:
- Support for multiple dataset types (acoustic, phone, speaker)
- Dynamic batch sizing with bucketing strategy
- Masked acoustic modeling (MAM) preprocessing
- Efficient data loading with configurable sampling

Author: fanfan-yu
Date: 2025.10.05
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from upstream.mam import process_train_MAM_data

##################
# CONSTANTS #
##################
# Half batch size threshold in time steps to prevent memory issues
HALF_BATCHSIZE_TIME = 3000
# Minimum number of utterances required for a speaker to be included
SPEAKER_THRESHOLD = 0

def load_libri_data(npy_path, npy_root=None):
    """
    Load LibriSpeech acoustic features from numpy files.
    
    This utility function loads pre-computed mel-spectrogram features stored as numpy arrays.
    It's designed for efficient loading of acoustic representations used in training.
    
    Args:
        npy_path (str): Relative path to the numpy file within the dataset root
        npy_root (str, optional): Root directory path for the dataset. If None, 
                                 defaults to the dataset-specific root path
    
    Returns:
        torch.FloatTensor: Loaded acoustic features as PyTorch tensor with shape (T, D)
                          where T is time steps and D is feature dimensions
    
    Raises:
        FileNotFoundError: If the specified numpy file doesn't exist
        ValueError: If the loaded data has incompatible dimensions
        
    Example:
        >>> features = load_libri_data('train-clean-100/103-1240-0001.npy', '/data/LibriSpeech')
        >>> print(features.shape)  # torch.Size([1000, 80])
    """
    return torch.FloatTensor(np.load(os.path.join(npy_root, npy_path)))

class BaseDataset(Dataset):
    """
    Base dataset class for Audio DistilBERT data loading with bucketing support.
    
    This abstract base class provides fundamental functionality for handling speech
    datasets with dynamic batching through bucketing strategy. It manages data
    sorting, filtering, and basic dataset operations.
    
    The bucketing strategy groups sequences of similar lengths together to maximize
    GPU utilization while preventing memory issues from extremely long sequences.
    
    Attributes:
        root (str): Root directory path for the dataset
        table (pd.DataFrame): Concatenated and sorted dataset information
        X (list): List of batched file paths for bucketing
    
    Args:
        file_path (str): Root directory containing dataset CSV files
        sets (list): List of dataset splits to load (e.g., ['train-clean-100'])
        bucket_size (int): Number of sequences per batch/bucket
        max_timestep (int, optional): Maximum sequence length in frames. 
                                    Sequences longer than this will be filtered if drop=True
        drop (bool, optional): Whether to drop sequences exceeding max_timestep
    
    Methods:
        __len__: Returns number of buckets in the dataset
        __getitem__: Abstract method to be implemented by subclasses
    
    Example:
        >>> dataset = BaseDataset('/data/LibriSpeech', ['train-clean-100'], 32, max_timestep=1000)
        >>> print(len(dataset))  # Number of buckets
    """
    
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False):
        """
        Initialize the base dataset.
        
        Args:
            file_path: Path to dataset directory
            sets: List of dataset splits
            bucket_size: Batch size per bucket
            max_timestep: Max sequence length filter
            drop: Whether to drop long sequences
        """
        # Read and process dataset metadata from CSV files
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)

        # Apply length filtering for computational efficiency
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]

    def __len__(self):
        """Return the number of buckets in the dataset."""
        return len(self.X)

class AcousticDataset(BaseDataset):

    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False, mam_config=None,
                 libri_root=None):
        super(AcousticDataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        self.mam_config = mam_config
        self.libri_root = libri_root
        self.sample_step = mam_config['max_input_length'] if 'max_input_length' in mam_config else 0
        if self.sample_step > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_step)

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)

            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_step > 0:
                    self.X.append(batch_x[:bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

    def sample(self, x):
        if len(x) < self.sample_step: return x
        idx = random.randint(0, len(x) - self.sample_step)
        return x[idx:idx + self.sample_step]

    def process_x_pad_batch(self, x_pad_batch):
        return process_train_MAM_data(spec=(x_pad_batch,), config=self.mam_config)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        if self.sample_step > 0:
            x_batch = [self.sample(load_libri_data(x_file, self.root)) for x_file
                       in self.X[index]]
        else:
            x_batch = [load_libri_data(x_file, self.root) for x_file in
                       self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return self.process_x_pad_batch(x_pad_batch)

#####################
# CPC PHONE DATASET #
#####################
'''
The LibriSpeech train-clean-100 (speech, phone) dataset, idendical alignment and split with the CPC paper
'''
class CPC_Phone_Dataset(BaseDataset):

    def __init__(self, file_path, phone_path, sets, bucket_size, max_timestep=0, drop=False, mam_config=None,
                 split='train', seed=1337,
                 libri_root=None, data_ratio=None):
        super(CPC_Phone_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        assert ('train-clean-100' in sets and len(sets) == 1)  # `sets` must be ['train-clean-100']
        random.seed(seed)
        self.mam_config = mam_config
        self.phone_path = phone_path
        self.libri_root = libri_root
        phone_file = open(os.path.join(phone_path, 'converted_aligned_phones.txt')).readlines()

        self.Y = {}
        # phone_set = []
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]
            # for p in line[1:]:
            # if p not in phone_set: phone_set.append(p)
        self.class_num = 41  # len(phone_set) # uncomment the above lines if you want to recompute

        if split == 'train' or split == 'dev':
            usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
            random.shuffle(usage_list)
            percent = int(len(usage_list) * 0.9)

            # use data_ratio
            old_percent = percent
            percent = int(percent * data_ratio)

            usage_list = usage_list[:percent] if split == 'train' else usage_list[old_percent:]
            # usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
        elif split == 'test':
            usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
        else:
            raise ValueError('Invalid \'split\' argument for dataset: CPC_Phone_Dataset!')
        usage_list = {line.strip('\n'): None for line in usage_list}
        print(
            '[Dataset] - Possible phone classes: ' + str(self.class_num) + ', number of data: ' + str(len(usage_list)))

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self.parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)

                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size // 2])
                        self.X.append(batch_x[bucket_size // 2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            if self.parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def match_sequence(self, x_batch, p_batch):
        scale = 1
        truncated_length = min(x_batch.shape[1] // scale, p_batch.shape[1])
        x_match_batch = x_batch[:, :truncated_length * scale, :]
        p_match_batch = p_batch[:, :truncated_length]
        return x_match_batch, p_match_batch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [load_libri_data(x_file, self.root) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        p_batch = [torch.LongTensor(self.Y[self.parse_x_name(x_file)]) for x_file in self.X[index]]
        p_pad_batch = pad_sequence(p_batch, batch_first=True)
        x_match_batch, p_match_batch = self.match_sequence(x_pad_batch, p_pad_batch)
        return x_match_batch, p_match_batch  # Return (x_spec, phone_label)


#######################
# MEL SPEAKER DATASET #
#######################
'''
The LibriSpeech (speech, speaker) dataset
'''


class Speaker_Dataset(Dataset):

    def __init__(self, split, file_path, sets, bucket_size, split_path=None, max_timestep=0, drop=False,
                 mam_config=None, seed=1337,
                 libri_root=None, data_ratio=1):
        random.seed(seed)
        self.mam_config = mam_config
        self.root = file_path
        self.libri_root = libri_root

        # Load the input sets
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Compute speaker dictionary
        print('[Dataset] - Computing speaker class...')
        speakers = self.get_all_speakers(X)
        self.speaker2idx = self.compute_speaker2idx(speakers)
        self.class_num = len(self.speaker2idx)

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]

        # if using 'train-clean-100' and the cpc split files exist, use them:
        usage_list = []
        if len(sets) == 1 and 'train-clean-100' in sets:
            # use CPC split:
            if (split == 'train' or split == 'dev') and os.path.isfile(os.path.join(split_path, 'train_split.txt')):
                usage_list = open(os.path.join(split_path, 'train_split.txt')).readlines()
                random.shuffle(usage_list)
                percent = int(len(usage_list) * 0.9)

                # use data_ratio
                old_percent = percent
                percent = int(percent * data_ratio)

                usage_list = usage_list[:percent] if split == 'train' else usage_list[old_percent:]
            elif split == 'test' and os.path.isfile(os.path.join(split_path, 'test_split.txt')):
                usage_list = open(os.path.join(split_path, 'test_split.txt')).readlines()
            else:
                raise NotImplementedError('Invalid `split` argument!')

            self.table = tables
            usage_list = {line.strip('\n'): None for line in usage_list}
            print('[Dataset] - Using CPC train/test splits.')
            print('[Dataset] - Possible speaker classes: ' + str(self.class_num) + ', number of data: ' + str(
                len(usage_list)))

        # else use random 8:1:1 split
        if len(usage_list) == 0:
            random.shuffle(X)
            percent_train, percent_dev, percent_test = int(len(X) * 0.8), int(len(X) * 0.1), int(len(X) * 0.1)
            if split == 'train':
                X = X[:percent_train]
            elif split == 'dev':
                X = X[percent_train: percent_train + percent_dev]
            elif split == 'test':
                X = X[-percent_test:]
            else:
                raise NotImplementedError('Invalid `split` argument!')
            print('[Dataset] - Possible speaker classes: ' + str(self.class_num) + ', number of data: ' + str(len(X)))

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if len(usage_list) == 0 or self.parse_x_name(x) in usage_list:  # check if x is in list if list not empty
                speaker = self.get_speaker_from_path(x)
                if speaker in self.speaker2idx:
                    batch_x.append(x)
                    batch_len.append(x_len)

                    # Fill in batch_x until batch is full
                    if len(batch_x) == bucket_size:
                        # Half the batch size if seq too long
                        if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                            self.X.append(batch_x[:bucket_size // 2])
                            self.X.append(batch_x[bucket_size // 2:])
                        else:
                            self.X.append(batch_x)
                        batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            if len(usage_list) == 0 or self.parse_x_name(x) in usage_list:  # check if x is in list if list not empty
                self.X.append(batch_x)

    def parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [load_libri_data(x_file, self.root) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        # Return (x_spec, speaker_label)
        s_batch = torch.LongTensor([self.speaker2idx[self.get_speaker_from_path(x_file)] for x_file in self.X[index]])
        return x_pad_batch, s_batch

    def get_speaker_from_path(self, x):
        return x.split('/')[-1].split('.')[0].split('-')[0]

    def get_all_speakers(self, X):
        speaker_set = {}
        for x in X:
            speaker = self.get_speaker_from_path(x)
            if speaker not in speaker_set:
                speaker_set[speaker] = 0
            else:
                speaker_set[speaker] += 1
        return speaker_set

    def compute_speaker2idx(self, speakers):
        idx = 0
        speaker2idx = {}
        for speaker in sorted(speakers):
            if speaker not in speaker2idx and speakers[
                speaker] > SPEAKER_THRESHOLD:  # eliminate the speakers with too few utterance
                speaker2idx[speaker] = idx
                idx += 1
        return speaker2idx


##################
# GET DATALOADER #
##################
def get_Dataloader(split, load, data_path, batch_size, max_timestep,
                   use_gpu, n_jobs, train_set, dev_set, test_set, dev_batch_size, phone_path=None, seed=1337,
                   mam_config=None, libri_root=None,
                   decode_beam_size=None, data_ratio=1, **kwargs):
    # Decide which split to use: train/dev/test
    if split == 'train':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    elif split == 'dev':
        bs = dev_batch_size
        shuffle = False
        sets = dev_set if load != 'cpc_phone' and load != 'speaker' else train_set  # the CPC paper uses its own train/test split from train-clean-100
        drop_too_long = True
    elif split == 'test':
        bs = 1 if decode_beam_size is not None else dev_batch_size
        n_jobs = 1
        shuffle = False
        sets = test_set if load != 'cpc_phone' and load != 'speaker' else train_set  # the CPC paper uses its own train/test split from train-clean-100
        drop_too_long = False
    else:
        raise NotImplementedError('Unsupported `split` argument: ' + split)

    # Decide which task (or dataset) to propogate through model
    if load == 'acoustic':
        ds = AcousticDataset(file_path=data_path, sets=sets, max_timestep=max_timestep,
                             bucket_size=bs, drop=drop_too_long, mam_config=mam_config,
                             libri_root=libri_root)
    elif load == 'cpc_phone':
        assert (phone_path is not None), '`phone path` must be provided for this dataset.'
        ds = CPC_Phone_Dataset(file_path=data_path, phone_path=phone_path, sets=sets, max_timestep=max_timestep,
                               bucket_size=bs, drop=drop_too_long, mam_config=mam_config, split=split, seed=seed,
                               libri_root=libri_root, data_ratio=data_ratio)
    elif load == 'speaker':
        ds = Speaker_Dataset(file_path=data_path, split_path=phone_path, sets=sets, max_timestep=max_timestep,
                             bucket_size=bs, drop=drop_too_long, mam_config=mam_config, split=split, seed=seed,
                             libri_root=libri_root, data_ratio=data_ratio)
    else:
        raise NotImplementedError('Invalid `load` argument for `get_Dataloader()`!')

    return DataLoader(ds, batch_size=1, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=use_gpu)
