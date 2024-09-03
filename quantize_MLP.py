import torch
import torch.nn as nn
import torch.optim as optim
from brevitas.nn import *
from brevitas.core.quant import QuantType
from brevitas.quant import Int8WeightPerTensorFixedPoint
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from coommon import *
# 設定隨機種子
torch.manual_seed(42)

# 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料預處理
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0  # 攤平成一維向量
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 將 NumPy 陣列轉換為 PyTorch 張量
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# 定義量化的 MLP 模型






class QuantizedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,quant_type = 'int',w_bit = 8,a_bit = 8):
        super(QuantizedMLP, self).__init__()
        
        # 使用 QuantLinear 和 QuantReLU 來進行量化#把QuantLinear 用 numpy 實現


        self.fc1 = QuantLinear(
            input_size, hidden_size,
            weight_quant=weight_quantizer[f'{quant_type}{w_bit}'],
            bias=True,
            return_quant_tensor=True
        )
        self.relu = QuantReLU(
            act_quant=act_quantizer[f'{quant_type}{a_bit}'],
            return_quant_tensor=True
        )
        self.fc2 = QuantLinear(
            hidden_size, output_size,
            weight_quant=weight_quantizer[f'{quant_type}{w_bit}'],
            bias=True,
            return_quant_tensor=True
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 超參數設定
input_size = x_train.shape[1]
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 10
batch_size = 32

model = QuantizedMLP(input_size, hidden_size, output_size)

# 定義損失函數和優化器
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
