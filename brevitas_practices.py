import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from abc import ABCMeta
from abc import abstractmethod
from functools import partial
import math
from typing import List, Optional, Tuple
from coommon import *
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from brevitas.nn import QuantRNN
import brevitas
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantSigmoid
from brevitas.nn import QuantTanh
from brevitas.nn.mixin import QuantBiasMixin
from brevitas.nn.mixin import QuantWeightMixin
from brevitas.nn.mixin.base import QuantRecurrentLayerMixin
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat

QuantTupleShortEnabled = List[Tuple[Tensor, Tensor, Tensor, Tensor]]
QuantTupleShortDisabled = List[Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]]
QuantTupleLongEnabled = List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
QuantTupleLongDisabled = List[Tuple[Tensor,
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor]]]

# 建立 MLP 模型
import numpy as np
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

# 設定隨機種子
np.random.seed(42)

# 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料預處理
x_train = x_train.reshape(x_train.shape[0], -1).astype('int8') / 255.0  # 攤平成一維向量
x_test = x_test.reshape(x_test.shape[0], -1).astype('int8') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 定義模型超參數
input_size = x_train.shape[1]  # 784
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 10
batch_size = 32

# 初始化權重和偏差
def initialize_weights(input_size, hidden_size, output_size):
    # 隨機生成浮點數權重（範圍選擇根據應用需求，這裡假設 [-0.1, 0.1）
    float_weights = np.random.uniform(-0.1, 0.1, size=hidden_size).astype(np.float32)
    quantizer = Int8WeightPerTensorFixedPoint(float_weights)
    W1 = weight_quantizer(float_weights)
    b1 = np.zeros((1, hidden_size))

    W2 = weight_quantizer(float_weights)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# 定義激活函數及其導數
def relu(Z):

    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# 向前傳播
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 計算損失
def compute_loss(Y, A2):
    m = Y.shape[0]
    log_likelihood = -np.log(A2[range(m), np.argmax(Y, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# 反向傳播
def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# 更新權重
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# 初始化權重
W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

# 訓練模型
for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batch_size):
        X_batch = x_train[i:i + batch_size]
        Y_batch = y_train[i:i + batch_size]
        
        # 向前傳播
        Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
        
        # 計算損失
        loss = compute_loss(Y_batch, A2)
        
        # 反向傳播
        dW1, db1, dW2, db2 = backward_propagation(X_batch, Y_batch, Z1, A1, Z2, A2, W2)
        
        # 更新權重
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    # 每個 epoch 的輸出損失
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# 測試模型
Z1, A1, Z2, A2 = forward_propagation(x_test, W1, b1, W2, b2)
predictions = np.argmax(A2, axis=1)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test accuracy: {accuracy:.4f}")
