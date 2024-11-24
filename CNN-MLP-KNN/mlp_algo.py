import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class DDIAKMLP:
    def __init__(self, input_size, hidden_sizes, output_size, lr=0.01, dropout_rate=0.5):
        self.w1 = np.random.randn(input_size, hidden_sizes[0])*np.sqrt(2/input_size)
        self.b1 = np.zeros(hidden_sizes[0])
        self.w2 = np.random.randn(hidden_sizes[0], hidden_sizes[1])*np.sqrt(2/hidden_sizes[0])
        self.b2 = np.zeros(hidden_sizes[1])
        self.w3 = np.random.randn(hidden_sizes[1], output_size)*np.sqrt(2/hidden_sizes[1])
        self.b3 = np.zeros(output_size)
        self.lr = lr
        self.dropout_rate = dropout_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def dropout(self, x):
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
        return x * mask / (1 - self.dropout_rate)

    def forward(self, x, train=True):
        self.h1 = self.relu(np.dot(x, self.w1) + self.b1)
        if train:
            self.h1 = self.dropout(self.h1)
        self.h2 = self.relu(np.dot(self.h1, self.w2) + self.b2)
        if train:
            self.h2 = self.dropout(self.h2)
        self.out = self.softmax(np.dot(self.h2, self.w3) + self.b3)
        return self.out

    def backward(self, x, y):
        o_err = self.out - y
        w3_grad = np.dot(self.h2.T, o_err)
        b3_grad = np.sum(o_err, axis=0)
        h2_err = np.dot(o_err, self.w3.T) * self.relu_derivative(self.h2)
        w2_grad = np.dot(self.h1.T, h2_err)
        b2_grad = np.sum(h2_err, axis=0)
        h1_err = np.dot(h2_err, self.w2.T) * self.relu_derivative(self.h1)
        w1_grad = np.dot(x.T, h1_err)
        b1_grad = np.sum(h1_err, axis=0)

        clip_threshold = 1.0
        w3_grad = np.clip(w3_grad, -clip_threshold, clip_threshold)
        w2_grad = np.clip(w2_grad, -clip_threshold, clip_threshold)
        w1_grad = np.clip(w1_grad, -clip_threshold, clip_threshold)

        self.w3 -= self.lr*w3_grad
        self.b3 -= self.lr*b3_grad
        self.w2 -= self.lr*w2_grad
        self.b2 -= self.lr*b2_grad
        self.w1 -= self.lr*w1_grad
        self.b1 -= self.lr*b1_grad

    def compute_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true*np.log(y_pred+1e-8), axis=1))

input_size=32*32*3
hidden_sizes=[256, 128] 
output_size=10
lr=0.01
dropout_rate=0.3
epochs=50
batch_size=64
model=DDIAKMLP(input_size, hidden_sizes, output_size, lr, dropout_rate)
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for e in range(epochs):
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    for i in range(0, x_train.shape[0], batch_size):
        xb = x_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]
        model.forward(xb, train=True)
        model.backward(xb, yb)

    train_pred = model.forward(x_train, train=False)
    test_pred = model.forward(x_test, train=False)
    train_loss = model.compute_loss(y_train, train_pred)
    test_loss = model.compute_loss(y_test, test_pred)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
    test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {e + 1}/{epochs} -----> Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("training_history_custom_mlp.png")
plt.close()

y_pred_labels = np.argmax(test_pred, axis=1)
true_labels = np.argmax(y_test, axis=1)
correct_indices = np.where(y_pred_labels == true_labels)[0][:7]
wrong_indices = np.where(y_pred_labels != true_labels)[0][:7]
fig, axes = plt.subplots(2, 7, figsize=(15, 6))
fig.suptitle('Correct and Incorrect Predictions', fontsize=16)

for i, idx in enumerate(correct_indices):
    ax = axes[0, i]
    ax.imshow(x_test[idx].reshape(32, 32, 3))
    ax.set_title(f"Pred: {y_pred_labels[idx]}, True: {true_labels[idx]}")
    ax.axis('off')
for i, idx in enumerate(wrong_indices):
    ax = axes[1, i]
    ax.imshow(x_test[idx].reshape(32, 32, 3))
    ax.set_title(f"Pred: {y_pred_labels[idx]}, True: {true_labels[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions_custom_mlp.png")
plt.show()

print(f"Final Test Accuracy: {test_acc:.4f}")
