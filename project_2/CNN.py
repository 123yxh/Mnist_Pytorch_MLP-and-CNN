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


# 定义CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道1（灰度图），输出通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道32，输出通道64
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层，输入64*7*7，输出128
        self.fc2 = nn.Linear(128, 10)  # 输出层，输出10个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积+激活+池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积+激活+池化
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层+激活
        return self.fc2(x)  # 输出层

class LightCNNWithMixedPooling(nn.Module):
    def __init__(self):
        super(LightCNNWithMixedPooling, self).__init__()
        # 第一层卷积：16个输出通道
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)  # 最大池化
        # 第二层卷积：32个输出通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.avgpool = nn.AvgPool2d(2, 2)  # 平均池化
        # 全连接层，设置输入为卷积和池化后展平的特征大小
        self.fc1 = nn.Linear(32 * 5 * 5, 64)  # 输入大小调整为32 * 5 * 5
        self.fc2 = nn.Linear(64, 10)          # 输出层

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))   # 第一层卷积 + ReLU激活 + 最大池化
        x = self.avgpool(torch.relu(self.conv2(x)))   # 第二层卷积 + ReLU激活 + 平均池化
        x = x.view(x.size(0), -1)                     # 展平特征图
        x = torch.relu(self.fc1(x))                   # 全连接层 + 激活函数
        return self.fc2(x)                            # 输出层



# 初始化模型、损失函数和优化器
# model = CNN()
model = LightCNNWithMixedPooling()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练并记录每轮的loss和accuracy
epochs = 100
train_losses = []
test_losses = []
test_accuracies = []

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
plt.title("Test Loss Epochs By CNN_2")
plt.legend()

# 子图2：绘制accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy", color="blue")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Test Accuracy Epochs By CNN_2")
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()
