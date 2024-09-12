
# 載入權重# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm
from brevitas.nn import *
from brevitas.core.quant import QuantType
from brevitas.quant import Int8WeightPerTensorFixedPoint
# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
class FI2010Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
# load packages
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import yaml
import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver

from brevitas.inject import ExtendedInjector
from brevitas.quant.base import *

from brevitas.quant.scaled_int import Int8WeightPerTensorFloat, \
    Int8ActPerTensorFloat, \
    Uint8ActPerTensorFloat, \
    Int8Bias, \
    Int16Bias, \
    Int32Bias
    

from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, \
    Int8ActPerTensorFixedPoint, \
    Uint8ActPerTensorFixedPoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0

class Int2WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width=2

class Int2ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width=2

class Uint2ActPerTensorFloat(Uint8ActPerTensorFloat):
    bit_width=2

class Int4WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width=4

class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width=4

class Uint4ActPerTensorFloat(Uint8ActPerTensorFloat):
    bit_width=4

class Int16ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width=16

class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width=4

class Int2WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width=2

class Int32ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width=32

class Int16ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width=16

class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width=4

class Int2ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width=2

class Uint4ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width=4

class Uint2ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width=2

weight_quantizer = {'int8': Int8WeightPerTensorFloat,
                    'int4': Int4WeightPerTensorFloat,
                    'int2': Int2WeightPerTensorFloat,
                    'fxp8': Int8WeightPerTensorFixedPoint,
                    'fxp4': Int4WeightPerTensorFixedPoint,
                    'fxp2': Int2WeightPerTensorFixedPoint}

act_quantizer = {
                'int16': Int16ActPerTensorFloat,
                'int8': Int8ActPerTensorFloat,
                'uint8': Uint8ActPerTensorFloat,
                'int4': Int4ActPerTensorFloat,
                'uint4': Uint4ActPerTensorFloat,
                'int2': Int2ActPerTensorFloat,
                'uint2': Uint2ActPerTensorFloat,
                'fxp32': Int32ActPerTensorFixedPoint,
                'fxp16': Int16ActPerTensorFixedPoint,
                'fxp8': Int8ActPerTensorFixedPoint,
                'fxp4': Int4ActPerTensorFixedPoint,
                'fxp2': Int2ActPerTensorFixedPoint,
                'ufxp8': Uint8ActPerTensorFixedPoint,
                'ufxp4': Uint4ActPerTensorFixedPoint,
                'ufxp2': Uint2ActPerTensorFixedPoint
                }

bias_quantizer = {'int8': Int8Bias,
                  'int16': Int16Bias,
                  'int32': Int32Bias}
import brevitas

class LoBMLP_Model(nn.Module):
    def __init__(self, input_dim):
        super(LoBMLP_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        quant_type = 'int'
        self.fc1 = QuantLinear(
                input_dim, 128,
                weight_quant=weight_quantizer[f'{quant_type}{8}'],
                bias=True,
                return_quant_tensor=True
            )
        self.relu1 = QuantReLU(
            act_quant=act_quantizer[f'{quant_type}{8}'],
            return_quant_tensor=True
        )
        self.fc2 = QuantLinear(
                128, 32,
                weight_quant=weight_quantizer[f'{quant_type}{8}'],
                bias=True,
                return_quant_tensor=True
            )
        self.relu2 = QuantReLU(
            act_quant=act_quantizer[f'{quant_type}{8}'],
            return_quant_tensor=True
        )
        self.fc3 = QuantLinear(
                32, 16,
                weight_quant=weight_quantizer[f'{quant_type}{8}'],
                bias=True,
                return_quant_tensor=True
            )
        self.relu3 = QuantReLU(
            act_quant=act_quantizer[f'{quant_type}{8}'],
            return_quant_tensor=True
        )
        self.fc4 = QuantLinear(
                16, 5,
                weight_quant=weight_quantizer[f'{quant_type}{8}'],
                bias=True,
                return_quant_tensor=True
            )
            
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
# For plotting learning curve
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[1:,-5:], valid_data[1:,-5:]
    raw_x_train, raw_x_valid, raw_x_test = train_data[1:,:-5], valid_data[1:,:-5], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(range(0,40)) # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'k': 16,
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 1000,     # Number of epochs.            
    'batch_size': 512, 
    'learning_rate': 1e-5,              
    'early_stop': 100,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
from torch.utils.tensorboard import SummaryWriter
# Set seed for reproducibility

checkpoint = torch.load('C:/Users/yuze/Desktop/lab_project/quantized_brevitas/model.ckpt', map_location=torch.device('cpu'))
# 查看檔案內容，通常包括模型的狀態、優化器狀態等

print(checkpoint.keys())

print((checkpoint['fc3.weight'][0]).dtype)
# model = LoBMLP_Model(input_dim=40)
# model.load_state_dict(model_weights)
# model.eval()
# 假設你有一個自定義的模型

# 如果你只想載入模型的權重
# model_weights = checkpoint['state_dict']
