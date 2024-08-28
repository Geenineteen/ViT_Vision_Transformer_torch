"""
Utility functions for setting up and managing a deep learning training environment using PyTorch.
"""
import os
import torch
import shutil
import numpy as np
from torch import nn
from model import (vit_base_patch16_224_in21k,
                   vit_base_patch32_224_in21k)


def set_seed(seed):
    """
    Sets the random seed to ensure reproducibility of results across runs.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):

    if args.model == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(args.num_classes, has_logits=False)
    else:
        raise Exception("Can't find any model name call {}".format(args.model))

    return model


def model_parallel(args, model):
    """
    Enables data parallelism for the given model using multiple GPUs.
    """

    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def remove_dir_and_create_dir(dir_name):
    """
    Removes the specified directory if it exists, and then creates a new directory 
    with the same name. This is useful for resetting the directory structure before training.

    Args:
        dir_name (str): The path of the directory to be removed and recreated.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")