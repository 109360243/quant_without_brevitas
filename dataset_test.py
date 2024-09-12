
from brevitas.quant import Int8WeightPerTensorFloat
import numpy as np
array = [0.1,0.2,0,3.3,0.4,0.5,0.6,0.7]
float_weight = np.array(array).astype('float32')

def quantiz (q_min,q_max,array):
    S = (max(array)-min(array))/(q_max-q_min)

quantizer = Int8WeightPerTensorFloat(
        quant_type=int,  # 使用整數量化
        bit_width=8,  # 8 位量化
        min_val=-0.5,  # 最小值
        max_val=0.5  # 最大值
    )

print(quantizer(float_weight))