"""
entry point of the system
"""
import os
import argparse
import logging
import sys

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from model.unet import Unet


def train(
        model,
        device,
        epochs=5,
        batch_size=1,
        learning_rate=1e-5
):
    # dataset =

    return


def parse_args():
    parser = argparse.ArgumentParser(description="Train the UNet on image data")


if __name__ == '__main__':
    train()
