
import numpy as np
from torch.nn import *
import torch
from typing import Optional, Type, Union
from torch import Tensor
from brevitas.quant_tensor import QuantTensor
from torch.nn.functional import linear
from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
#用來放置 MLP_qunwithoutbrevitas 這個 py 檔案可能會用到的 quant 函數

class Quant():#透過 r = S(q-z)去量化
    def __init__(self,qmin,qmax,rmin,rmax,r):
        super(Quant, self).__init__
    #透過 r = S(q-z)去實作量化 v
    def quant_impl(self,qmin,qmax,rmin,rmax,r):
        S = (rmax - rmin) / (qmax - qmin)
        z = qmin - rmin/S
        
        for i in range(0, r.shape[0]):
            r[i] = r[i]/S+z
        return r
test_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
qmin = -8
qmax = 7
rmin = min(test_list)
rmax = max(test_list)
test_list_np = np.array(test_list).astype('float32')
qu  = Quant(qmin,qmax,rmin,rmax,test_list_np)
class Int8WeightPerTensorFloat_impl():
     qu.quant_impl(qmin,qmax,rmin,rmax,test_list_np).astype('int8')
    

class Quant_linear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias:  True,
        weight_quant: Int8WeightPerTensorFloat_impl,
        bias_quant:  None,
        input_quant: None,
        output_quant: None,
        return_quant_tensor: bool = False,
        device  = None,
        dtype = None,
        **kwargs) -> None:
     Linear.__init__(self, in_features, out_features, bias, device=device, dtype=dtype)
    @property
    def per_elem_ops(self):
        return 2 * self.in_features
    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.out_features

    @property
    def channelwise_separable(self) -> bool:
        return False

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        output_tensor = linear(x, quant_weight, quant_bias)#做向前傳播的動作 
        return output_tensor

    def quant_output_scale_impl(
            self, inp: Tensor, quant_input_scale: Tensor, quant_weight_scale: Tensor):
        if quant_input_scale.shape == ():#看 quant_input_scale 是不是一個標量``
            input_broadcast_shape = tuple([1] * len(inp.size()))
            quant_input_scale = quant_input_scale.view(input_broadcast_shape) # 改變向量的形狀
        if quant_weight_scale.shape == ():
            weight_broadcast_shape = tuple([1] * len(self.weight.size()))
            quant_weight_scale = quant_weight_scale.view(weight_broadcast_shape)
        quant_output_scale = linear(quant_input_scale, quant_weight_scale)
        return quant_output_scale

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):#这个累加比特宽度是确保计算过程中不发生溢出所需的最小比特宽度
        max_input_val = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_fc_val = self.weight_quant.max_uint_value(weight_bit_width)#用于计算权重在指定 weight_bit_width 下的最大无符号整数值
        max_output_val = max_input_val * max_fc_val * self.in_features
        output_bit_width = ceil_ste(torch.log2(max_output_val))#这一步是为了找到用二进制表示 max_output_val 需要的位数，也就是是保存 max_output_val 所需的最小比特宽度。
        return output_bit_width


    