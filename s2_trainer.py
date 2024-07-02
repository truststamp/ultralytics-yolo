import os
import random
import sys
import time
from tempfile import TemporaryDirectory

sys.path.insert(0, '/home/luke/code/ultralytics')


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ultralytics import YOLO


def main():
    model = YOLO("/home/luke/code/ship_detection/yolov8n.yaml", task='detect_s2')
    model.train(data="/home/luke/code/ship_detection/dataset.yaml", epochs=100, 
                plots=True,
                # device='cpu',
                amp=False,
                imgsz=1024,
                single_cls = True,
                batch = 4,
                workers = 4,
                augment=False,
                val=True,
                rect=False,
                )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()