import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from torch_poly_lr_decay import PolynomialLRDecay
import random

# 병렬로 사용할 수
torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# labels 에는 답
labels_df = pd.read_csv('../dacon_data/dirty_mnist_answer.csv')[:]
imgs_dir = np.array(sorted(glob.glob('../dacon_data/dirty/train/*')))[:]

print(imgs_dir[9995:10005])