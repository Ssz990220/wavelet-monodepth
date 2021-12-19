import matplotlib.pyplot as plt
import cv2 as cv
import requests
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from pytorch_wavelets import IDWT
from networks.encoders import *
from phone_cam_utils_NYU import *
import matplotlib.gridspec as gridspec

url = "http://10.42.0.171:8080/shot.jpg"
device = "cuda"
threshold = 0.1
split = 1
kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
kernel = kernel/np.sum(kernel)
def main():   
    sparse_model=build_model()
    cv.namedWindow("Display", cv.WINDOW_AUTOSIZE)

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img_o = cv.imdecode(img_arr, -1)
        if split:
            img = cv.resize(img_o,(1280,480),interpolation=cv.INTER_NEAREST)
            img1 = img[:,0:640,:]
            img2 = img[:,640:,:]
            output1 = estimate(img1, sparse_model)
            output2 = estimate(img2, sparse_model)
            output_cmap = match_depth(output1, output2)
        else:
            img = cv.resize(img_o,(640,480),interpolation=cv.INTER_NEAREST)
            output = estimate(img, sparse_model)
            output = (output*(255.0/output.max())).astype(np.uint8)
            output_cmap = cv.applyColorMap(output,cv.COLORMAP_JET)
            img = cv.resize(img_o,(1280,480),interpolation=cv.INTER_NEAREST)
        output_cmap = cv.resize(output_cmap,(1280,480),interpolation=cv.INTER_NEAREST)
        output_cmap = cv.filter2D(output_cmap,-1,kernel)
        vis = np.concatenate((output_cmap,img), axis=0)
        cv.imshow("Display", vis)
        if cv.waitKey(1) == 27:
            break
    cv.desktroyAllWIndows()

def estimate(img, model):
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
    avg1 = np.average(output1[:,-3:-1])
    avg2 = np.average(output2[:,0:3])
    output2 = output2 + (avg1-avg2)
    output = np.concatenate((output1,output2),axis=1)
    output = (255.0-output*(255.0/output.max())).astype(np.uint8)
    output_cmap = cv.applyColorMap(output,cv.COLORMAP_JET)
    return output_cmap

main()