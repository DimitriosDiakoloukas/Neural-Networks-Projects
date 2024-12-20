import numpy as np
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from libsvm.svmutil import *
import time
from load_data_cifer import load_cifar10_data

def perform_pca_specific_variance(data, n_components):
    print("Performing PCA with specific variance...")
    pca = PCA(n_components)
    transformed_data = pca.fit_transform(data)
    print(f"Number of principal components: {pca.n_components_}")
    return transformed_data

def train_svm_with_libsvm(x_train, y_train, params):
    print("Training SVM with libsvm...")
    if params is None:
        params = '-t 0'  # Default linear kernel
    svm_model = svm_train(y_train.tolist(), x_train.tolist(), params)
    return svm_model

def evaluate_svm_with_libsvm(svm_model, x_test, y_test):
    print("Evaluating SVM...")
    y_pred, _, _ = svm_predict(y_test.tolist(), x_test.tolist(), svm_model)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def plot_results(y_test, y_pred, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

data_dir = 'cifar-10-batches-py'
num_classes = 10

(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

X = np.vstack((x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
y = np.hstack((y_train, y_test))

pca_variance = 0.90
print(f"Performing PCA to retain {pca_variance * 100}% variance")
X_pca = perform_pca_specific_variance(X, pca_variance)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

start_time = time.time()
svm_model = train_svm_with_libsvm(X_train, y_train, '-t 0')
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

start_time = time.time()
accuracy, y_pred = evaluate_svm_with_libsvm(svm_model, X_test, y_test)
evaluation_time = time.time() - start_time

print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

plot_results(y_test, y_pred, accuracy)