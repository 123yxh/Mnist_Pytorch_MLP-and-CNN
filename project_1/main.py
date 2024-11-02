import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
input_size = 28 * 28  # MNIST 图片 28x28 像素
output_size = 10  # 10 类别
num_epochs = 50 # 训练50次
hidden_size = 12 # MLP隐藏层数量
batch_size = 64
learning_rate = 0.01

# 数据集加载和预处理---Mnist数据集默认[0，1]之间
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 单层神经网络
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size)  # 展平输入
        return self.fc(x)

# MLP----hidden_size个隐藏层，激活函数为ReLU，更好的捕捉图像特征
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size)  # 展平输入
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积和池化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第二层卷积和池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)

        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))

        # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 5 * 5)

        # 全连接层 + 激活
        x = F.relu(self.fc1(x))

        # 输出层
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # 第一层卷积，使用 5x5 卷积核，输出通道 32，步长 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # 保持输入尺寸
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 使用平均池化

        # 第二层卷积，使用 5x5 卷积核，输出通道 64，步长 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 使用最大池化

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 更新展平后的输入尺寸
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 + 激活 + 平均池化
        x = self.pool1(F.relu(self.conv1(x)))

        # 第二层卷积 + 激活 + 最大池化
        x = self.pool2(F.relu(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层 + 激活
        x = F.relu(self.fc1(x))

        # 输出层
        x = self.fc2(x)
        return x

# 初始化模型、损失函数-交叉损失函数和优化器-随机梯度下降
# model = SingleLayerNN(input_size, output_size).to(device)
# 多层感知机
# model = MLP(input_size, hidden_size, output_size).to(device)
# CNN model
model = CNN2().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 记录训练过程的损失和准确率
test_losses = []
test_accuracies = []


# 定义测试函数
def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 移动数据到 GPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / len(test_dataset)
    return test_loss, accuracy


# 打开文件用于写入loss和accuracy
with open("cnn2_results.txt", "w") as f:
    f.write("Epoch\tTest Loss\tTest Accuracy\n")  # 写入表头

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 移动数据到 GPU

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每轮训练结束后进行测试
        test_loss, test_accuracy = evaluate()
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # 将结果写入文件
        f.write(f"{epoch + 1}\t{test_loss:.4f}\t{test_accuracy:.2f}\n")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# 绘制损失和准确率曲线
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
