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