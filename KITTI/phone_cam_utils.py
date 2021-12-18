import os
import time
import matplotlib.pyplot as plt
# %matplotlib notebook

from collections import OrderedDict
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.models as torch_models
import torch.utils.model_zoo as model_zoo

from PIL import Image
from torchvision import transforms as T

from pytorch_wavelets import IDWT

from networks.decoders import DepthWaveProgressiveDecoder, SparseDepthWaveProgressiveDecoder
from networks.encoders import *


model_path = "./KITTI/weights/HR_Res50"
# Encoder Parameters
num_layers = 50
# Decoder Parameters
output_scales = [0, 1, 2, 3]
sparse_scales = [0, 1, 2, 3]
threshold = 0.0
device = "cuda"

class DenseModel(nn.Module):
    def __init__(self, num_layers, output_scales, device="cpu"):
        super(DenseModel, self).__init__()
        device = torch.device("cpu" if device=="cpu" else "cuda")
        self.models = {}
        self.models["encoder"] = ResnetEncoder(num_layers, False)
        self.models["depth"] = DepthWaveProgressiveDecoder(
                        self.models["encoder"].num_ch_enc, scales=output_scales)       
        
        self.models["encoder"].to(device)
        self.models["depth"].to(device)
    
    def forward(self, x):
        features_encoder = self.models["encoder"](x)
        outputs = self.models["depth"](features_encoder)
        return outputs

    
class SparseModel(nn.Module):
    def __init__(self, num_layers, output_scales, sparse_scales, device="cpu"):
        super(SparseModel, self).__init__()
        device = torch.device("cpu" if device=="cpu" else "cuda")
        self.models = {}
        self.models["encoder"] = ResnetEncoder(num_layers, False)
        self.models["depth"] = SparseDepthWaveProgressiveDecoder(
                        self.models["encoder"].num_ch_enc, scales=output_scales)       
        
        self.models["encoder"].to(device)
        self.models["depth"].to(device)
        self.sparse_scales = sparse_scales
    
    def forward(self, x, thresh_ratio):
        features_encoder = self.models["encoder"](x)
        outputs = self.models["depth"](features_encoder, thresh_ratio, self.sparse_scales)
        return outputs

def build_model():
    sparse_model = SparseModel(num_layers, output_scales, sparse_scales, device=device)
    sparse_model.eval()
    load_model(sparse_model, model_path)
    sparse_model.models["encoder"].eval()
    return sparse_model

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def load_model(model, load_weights_folder):
    """Load model(s) from disk
    """
    models_to_load = ["encoder", "depth"]
    load_weights_folder = os.path.expanduser(load_weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))    

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = model.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location={"cuda:0": "cpu"})
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.models[n].load_state_dict(model_dict)

def to_torch(img):
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)
    return img_tensor

def set_left_title(title):
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
    plt.tick_params(axis = "y", which = "both", left = False, right = False)
    plt.ylabel(title)
    plt.box(on=None)