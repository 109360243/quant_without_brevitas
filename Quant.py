
import numpy as np
from torch.nn import *


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

    