import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from pytorch_wavelets import IDWT
from networks.decoders import SparseDecoderWave, DecoderWave, Decoder
from networks.encoders import *
    
class SparseModel(nn.Module):
    def __init__(self, opts):
        super(SparseModel, self).__init__()
        
        decoder_width = 0.5
        device = "cuda"
        self.encoder = DenseEncoder(normalize_input=opts.normalize_input, pretrained=opts.pretrained_encoder).to(device)    
        self.decoder = SparseDecoderWave(enc_features=self.encoder.num_ch_enc, decoder_width=decoder_width).to(device)

    def forward(self, x, thresh_ratio=0.1):
        return self.decoder( self.encoder(x), thresh_ratio)

class Options(object):
    def __init__(self):
        super(Options, self).__init__()

def build_model():
    opts = Options()
    opts.encoder_type = "densenet"
    opts.output_scales =  [0, 1, 2, 3]  
    opts.normalize_input = True
    opts.use_wavelets = True      
    opts.pretrained_encoder = False
    sparse_model = SparseModel(opts)
    sparse_model = load_model(sparse_model, "./NYUv2/weights/")
    sparse_model.eval()
    return sparse_model

def load_model(model, load_weights_folder):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(load_weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))    

    n = "model"
    print("Loading {} weights...".format(n))
    path = os.path.join(load_weights_folder, "{}.pth".format(n))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location={"cuda:0": "cpu"})
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def to_torch(img):
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)
    return img_tensor 