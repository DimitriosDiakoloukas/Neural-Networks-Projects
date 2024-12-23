import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from libsvm.svmutil import *
import time
from load_data_cifer import load_cifar10_data

def train_svm_with_libsvm(x_train, y_train, params):
    print("Training SVM with libsvm...")
    svm_model = svm_train(y_train.tolist(), x_train.tolist(), params)
    return svm_model

def evaluate_svm_with_libsvm(svm_model, x_test, y_test):
    print("Evaluating SVM...")
    y_pred, _, _ = svm_predict(y_test.tolist(), x_test.tolist(), svm_model)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def plot_results(y_test, y_pred, accuracy, kernel_type):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Kernel: {kernel_type}, Accuracy: {accuracy:.4f})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plot_file = f"confusion_matrix_2_classes_{kernel_type}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Confusion matrix saved as {plot_file}")

data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

X = np.vstack((x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
y = np.hstack((y_train, y_test))

selected_classes = [0, 1]
filter_indices = np.isin(y, selected_classes)
X_filtered = X[filter_indices]
y_filtered = y[filter_indices]

y_filtered = np.where(y_filtered == selected_classes[0], 0, 1)

sample_indices = np.random.choice(X_filtered.shape[0], 10000, replace=False)
X_sampled = X_filtered[sample_indices]
y_sampled = y_filtered[sample_indices]

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.4, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

results_file = "svm_results_2_classes.txt"
with open(results_file, "w") as f:
    kernels = {'linear': '-t 0', 'polynomial': '-t 1', 'radial basis function': '-t 2', 'sigmoid': '-t 3'}

    for kernel_name, kernel_param in kernels.items():
        f.write(f"\nUsing {kernel_name} kernel\n")
        print(f"\nUsing {kernel_name} kernel")
        start_time = time.time()
        svm_model = train_svm_with_libsvm(X_train, y_train, kernel_param)
        training_time = time.time() - start_time
        f.write(f"Training time for {kernel_name} kernel: {training_time:.2f} seconds\n")
        print(f"Training time for {kernel_name} kernel: {training_time:.2f} seconds")

        start_time = time.time()
        accuracy, y_pred = evaluate_svm_with_libsvm(svm_model, X_test, y_test)
        evaluation_time = time.time() - start_time

        f.write(f"Evaluation time for {kernel_name} kernel: {evaluation_time:.2f} seconds\n")
        f.write(f"Accuracy for {kernel_name} kernel: {accuracy:.4f}\n")
        print(f"Evaluation time for {kernel_name} kernel: {evaluation_time:.2f} seconds")
        print(f"Accuracy for {kernel_name} kernel: {accuracy:.4f}")

        report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
        f.write(f"Classification Report for {kernel_name} kernel:\n{report}\n")
        print(f"Classification Report for {kernel_name} kernel:\n", report)

        plot_results(y_test, y_pred, accuracy, kernel_name)

print(f"Results saved to {results_file}")
