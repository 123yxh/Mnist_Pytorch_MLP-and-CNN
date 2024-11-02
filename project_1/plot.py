import matplotlib.pyplot as plt

# 读取数据
def read_data(filename):
    epochs, losses, accuracies = [], [], []
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            epoch, loss, accuracy = line.strip().split('\t')
            epochs.append(int(epoch))
            losses.append(float(loss))
            accuracies.append(float(accuracy))
    return epochs, losses, accuracies

# 读取四个不同model的loss,accuracy数据
epochs1, losses1, accuracies1 = read_data('signal_results.txt')
epochs2, losses2, accuracies2 = read_data('mlp_results.txt')
epochs3, losses3, accuracies3 = read_data('cnn1_results.txt')
epochs4, losses4, accuracies4 = read_data('cnn2_results.txt')

# Loss 图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs1, losses1, label="SNN")
plt.plot(epochs2, losses2, label="MLP")
plt.plot(epochs3, losses3, label="CNN1")
plt.plot(epochs4, losses4, label="CNN2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison Across Models")
plt.legend()

# Accuracy 图
plt.subplot(1, 2, 2)
plt.plot(epochs1, accuracies1, label="SNN")
plt.plot(epochs2, accuracies2, label="MLP")
plt.plot(epochs3, accuracies3, label="CNN1")
plt.plot(epochs4, accuracies4, label="CNN2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison Across Methods")
plt.legend()

plt.tight_layout()
plt.show()
