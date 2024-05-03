import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# 加载数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root='Datasets/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='Datasets/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
#访问样本
feature, label = mnist_train[0]
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

num_inputs = 784
X = feature.view(-1, num_inputs)

# 定义数据加载器
batch_size = 256
num_workers = 0
train_loader = Data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=False )
test_loader = Data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False,num_workers=num_workers,drop_last=False)

def train(train_loader, num_epochs, lr=0.1):
    n_features = feature.view(-1, num_inputs).shape[1]
    n_classes = 10
    W = torch.randn(n_classes, n_features)
    b = torch.zeros(n_classes, 1)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X = X.view(X.shape[0], -1)
            y_onehot = F.one_hot(y, num_classes=n_classes).float().T

            O = torch.mm(W, X.t()) + b
            O -= torch.max(O, dim=0)[0]  # 稳定softmax计算
            y_hat = F.softmax(O, dim=0)
            
            loss = -torch.sum(y_onehot * torch.log(y_hat + 1e-7)) / len(y_onehot)
            total_loss += loss
            
            m = len(y)
            dW = 1/m * torch.mm((y_hat - y_onehot), X)
            db = 1/m * torch.sum(y_hat - y_onehot, axis=1, keepdim=True)

            W -= lr * dW
            b -= lr * db
            
            _, predicted = torch.max(y_hat, 0)
            correct += (predicted == y).sum().item()
            total += len(y)
        
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    return W,b

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def test(test_loader, W, b):
    n_classes = W.shape[0]
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    for X, y in test_loader:
        X = X.view(X.shape[0], -1)
        O = torch.mm(W, X.t()) + b
        O -= torch.max(O, dim=0)[0]  # 稳定softmax计算
        y_hat = F.softmax(O, dim=0)
        
        _, predicted = torch.max(y_hat, 0)
        correct += (predicted == y).sum().item()
        total += len(y)
        
        y_true.extend(y.tolist())
        y_pred.extend(predicted.tolist())
    
    accuracy = correct / total
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

# 调用训练函数进行模型训练
num_epochs = 10
W,b=train(train_loader, num_epochs)
test(test_loader,W,b)

