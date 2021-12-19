import os
import time
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import requests
import timeit
from collections import OrderedDict
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
from phone_cam_utils import *
import matplotlib.gridspec as gridspec

url = "http://10.42.0.171:8080/shot.jpg"
split = 1

def main():   
    sparse_model=build_model()
    cv.namedWindow("Display", cv.WINDOW_AUTOSIZE)
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv.imdecode(img_arr, -1)
        if split:
            img = cv.resize(img,(1024,640),interpolation=cv.INTER_CUBIC)
            img1 = img[0:319,:,:]
            img2 = img[319:639,:,:]
            output1 = estimate(img1,sparse_model)
            output2 = estimate(img2,sparse_model)
            output_cmap = match_depth(output1,output2)
        else:
            img = cv.resize(img,(1024,320),cv.INTER_NEAREST)
            output = estimate(img,sparse_model)
            output = (output*(255.0/output.max())).astype(np.uint8)
            output_cmap = cv.applyColorMap(output,cv.COLORMAP_JET)
        vis = np.concatenate((output_cmap, img), axis=0)
        cv.imshow("Display", vis)
        if cv.waitKey(1) == 27:
            break
    cv.desktroyAllWIndows()

def estimate(img,model):
    img_tensor = to_torch(img)
    if device=="cuda":
        img_tensor = img_tensor.cuda()
    # Inspect
    with torch.no_grad():
        sparse_outputs = model(img_tensor, thresh_ratio=threshold)
    if device == "cuda":
        output = sparse_outputs[('disp', 0)][0,0].cpu().numpy()/100
    else:
        output = sparse_outputs[('disp', 0)][0,0].numpy()/100
    return output

def match_depth(output1,output2):
    avg1 = np.average(output1[-1,:])
    avg2 = np.average(output2[0,:])
    output2 = output2 + (avg1-avg2)
    output = np.concatenate((output1,output2),axis=0)
    output = (output*(255.0/output.max())).astype(np.uint8)
    output_cmap = cv.applyColorMap(output,cv.COLORMAP_JET)
    return output_cmap

main()