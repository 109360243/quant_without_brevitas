from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from coommon import *
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(42)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0  # 攤平成一維向量
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 將 NumPy 陣列轉換為 PyTorch 張量
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)




# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)