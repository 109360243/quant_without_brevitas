import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.quant import Int32Bias
from coommon import *
# 載入 CIFAR-10 數據集
(train_X, train_y), (test_X, test_y) = cifar10.load_data()

# 資料預處理
train_X = train_X.astype('float32') / 255
test_X = test_X.astype('float32') / 255
train_y2 = to_categorical(train_y)
test_y2 = to_categorical(test_y)

# 將數據轉換為 PyTorch 張量並調整形狀
x_train = torch.tensor(train_X).permute(0, 3, 1, 2)  # 將形狀從 [batch_size, height, width, channels] 轉換為 [batch_size, channels, height, width]
x_test = torch.tensor(test_X).permute(0, 3, 1, 2)
y_train = torch.tensor(train_y2)
y_test = torch.tensor(test_y2)

class QuantizedCONV(nn.Module):
    def __init__(self, input_size, hidden_size_1, output_size, quant_type='int', w_bit=8, a_bit=8):
        super(QuantizedCONV, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=a_bit,return_quant_tensor=True)
        self.conv1 = QuantConv2d(3, 6, 5, bias=True, weight_bit_width=w_bit,bias_quant=Int32Bias)
        self.relu1 = QuantReLU(bit_width=a_bit, return_quant_tensor=True)
        self.conv2 = QuantConv2d(6, 16, 5, bias=True, weight_bit_width=w_bit,bias_quant=Int32Bias)
        self.relu2 = QuantReLU(bit_width=a_bit, return_quant_tensor=True)
        self.fc1 = QuantLinear(
            16*5*5, 120,
            weight_quant=weight_quantizer[f'{quant_type}{w_bit}'],
            weight_bit_width = 8,
            bias=True,
            bias_quant = Int32Bias,
            
        )
        self.relu3 = QuantReLU(bit_width=a_bit, return_quant_tensor=True)
        self.fc2   = QuantLinear(
            120, 10,             
            weight_quant=weight_quantizer[f'{quant_type}{w_bit}'],
            weight_bit_width = 8,
            bias=True,
            bias_quant = Int32Bias,)
        
    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        return out

# 設定參數
learning_rate = 0.01
input_size = 16 * 5 * 5
hidden_size_1 = 120
output_size = 10
epochs = 10
batch_size = 32

# 初始化模型
model = QuantizedCONV(input_size, hidden_size_1, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(epochs):
    model.train()
    for i in range(0, x_train.shape[0], batch_size):
        X_batch = x_train[i:i + batch_size]
        Y_batch = y_train[i:i + batch_size]
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 向前傳播
        outputs = model(X_batch)
        
        # 計算損失
        loss = criterion(outputs, torch.max(Y_batch, 1)[1])
        
        # 反向傳播
        loss.backward()
        
        # 更新權重
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 測試模型
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == torch.max(y_test, 1)[1]).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")
