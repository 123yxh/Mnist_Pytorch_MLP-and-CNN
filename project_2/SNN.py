import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据预处理，将图像缩放到[0,1]之间
transform = transforms.Compose([transforms.ToTensor()])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 单层神经网络--SNN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

# MLP--2个隐藏层，激活函数ReLU，Leaky ReLU
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # 第一层，256个神经元
        self.fc2 = nn.Linear(256, 128)      # 第二层，128个神经元
        self.fc3 = nn.Linear(128, 10)       # 输出层
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)  # 设置Leaky ReLU的负斜率

    def forward(self, x):
        x = x.view(-1, 28 * 28)          # 展平输入图像
        x = self.relu(self.fc1(x))       # 第一层激活，ReLU
        x = self.leaky_relu(self.fc2(x)) # 第二层激活，Leaky ReLU
        return self.fc3(x)               # 输出层

# 初始化模型、损失函数和优化器
# model = SimpleNN()
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练并记录每轮的loss和accuracy
epochs = 100
train_losses = []
test_losses = []
test_accuracies = []

# 开始训练
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 计算训练集的平均损失
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 计算测试集上的loss和accuracy
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 打印当前轮次的结果
    print(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 绘制loss和accuracy曲线，分成两个子图
plt.figure(figsize=(12, 6))

# 子图1：绘制loss曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color="red")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Test Loss Epochs By MLP")
plt.legend()

# 子图2：绘制accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy", color="yellow")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Test Accuracy Epochs By MLP")
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()

