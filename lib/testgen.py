import argparse

import numpy as np

import os

import shutil

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import warnings

from lib.dataset import MegaDepthDataset
from lib.exceptions import NoGradientError
from lib.loss import loss_function
from lib.model import D2Net


training_dataset = MegaDepthDataset(
    scene_list_path='megadepth_utils/train_scenes.txt',
    preprocessing='caffe'
)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=64,
    num_workers=1
)

data=[]
for batch_ndx, sample in enumerate(loader[:100]):


    data.append(sample)
    

import pickle

writefile = open("/cluster/scratch/tianhu/test/testgen.pkl", "wb")
pickle.dump(data, writefile)
writefile.close()

