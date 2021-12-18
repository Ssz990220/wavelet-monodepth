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

url = "http://192.168.3.102:8080/shot.jpg"
device = "cuda"
threshold = 0.05

def main():   
    sparse_model=build_model()
    cv.namedWindow("Display", cv.WINDOW_AUTOSIZE)

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv.imdecode(img_arr, -1)
        img = cv.resize(img,(640,480),interpolation=cv.INTER_NEAREST)
        img_tensor = to_torch(img)
        if device=="cuda":
            img_tensor = img_tensor.cuda()
        # Inspect
        with torch.no_grad():
            sparse_outputs = sparse_model(img_tensor, thresh_ratio=threshold)
        if device == "cuda":
            output = sparse_outputs[('disp', 0)][0,0].cpu().numpy()/100
        else:
            output = sparse_outputs[('disp', 0)][0,0].numpy()/100
        output = (output*(255.0/output.max())).astype(np.uint8)
        output_cmap = cv.applyColorMap(output,cv.COLORMAP_JET)
        img = cv.resize(img,(320,240),interpolation=cv.INTER_NEAREST)
        vis = np.concatenate((output_cmap,img), axis=0)
        cv.imshow("Display", vis)
        if cv.waitKey(1) == 27:
            break
    cv.desktroyAllWIndows()


main()