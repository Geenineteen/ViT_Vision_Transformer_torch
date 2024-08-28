"""
This script configures and parses command-line arguments for training a Vision Transformer (ViT) model.

The script sets up various hyperparameters, paths for datasets and results, GPU configuration, and other training options such as freezing layers and using pretrained weights. The arguments are parsed and used to control the behavior of the training script.
"""

import argparse
import os

parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.01)

# Path to dataset and results
parser.add_argument('--dataset_train_dir', type=str,
                    default="./path/to/train/",
                    help='The directory containing the train data.')
parser.add_argument('--dataset_val_dir', type=str,
                    default="./path/to/validation/",
                    help='The directory containing the val data.')
parser.add_argument('--summary_dir', type=str, default="./summary/vit_base_patch16_224",
                    help='The directory of saving weights and tensorboard.')

# Path to pretrained weight
parser.add_argument('--weights', type=str, default='./pretrain_weights/vit_base_patch16_224_in21k.pth',
                    help='Initial weights path.')

parser.add_argument('--freeze_layers', type=bool, default=True)
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='Select gpu device.')

parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                    help='The name of ViT model, Select one to train.')

# Define 5 class types
parser.add_argument('--label_name', type=list, default=[
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
], help='The name of class.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
