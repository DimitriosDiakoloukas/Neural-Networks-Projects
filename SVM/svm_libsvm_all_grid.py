import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from libsvm.svmutil import *
import matplotlib.pyplot as plt
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
    svm_model = svm_train(y_train.tolist(), x_train.tolist(), params)
    return svm_model

def evaluate_svm_with_libsvm(svm_model, x_test, y_test):
    print("Evaluating SVM...")
    y_pred, _, _ = svm_predict(y_test.tolist(), x_test.tolist(), svm_model)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def grid_search_libsvm_all_kernels(X_train, y_train):
    print("Performing grid search for libsvm across all kernels...")
    best_params = {}
    kernels = {'linear': 0, 'polynomial': 1, 'RBF': 2, 'sigmoid': 3}
    C_values = [0.1, 1, 10]
    gamma_values = [0.001, 0.01, 0.1, 1]

    for kernel_name, kernel_type in kernels.items():
        best_kernel_params = None
        best_kernel_accuracy = 0
        print(f"Grid search for {kernel_name} kernel...")
        for C in C_values:
            for gamma in gamma_values:
                params = f"-t {kernel_type} -c {C} -g {gamma}"
                svm_model = train_svm_with_libsvm(X_train, y_train, params)
                _, y_pred = evaluate_svm_with_libsvm(svm_model, X_train, y_train)
                accuracy = accuracy_score(y_train, y_pred)
                if accuracy > best_kernel_accuracy:
                    best_kernel_accuracy = accuracy
                    best_kernel_params = params

        best_params[kernel_name] = (best_kernel_params, best_kernel_accuracy)
        print(f"Best params for {kernel_name} kernel: {best_kernel_params}, Best training accuracy: {best_kernel_accuracy:.4f}")

    return best_params

def train_knn(X_train, y_train, n_neighbors=3):
    print("Training k-Nearest Neighbors (k-NN)...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")
    return accuracy, y_pred

def plot_results(y_test, y_pred, accuracy, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({model_name}, Accuracy: {accuracy:.4f})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plot_file = f"grid_confusion_matrix_{model_name}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Confusion matrix saved as {plot_file}")

data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

num_classes = 10
y_train = np.array(y_train)
y_test = np.array(y_test)

X = np.vstack((x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
y = np.hstack((y_train, y_test))

pca_variance = 0.90
X_pca = perform_pca_specific_variance(X, pca_variance)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)

best_params_all_kernels = grid_search_libsvm_all_kernels(X_train, y_train)

results_file = "grid_svm_knn_results.txt"
with open(results_file, "w") as f:
    for kernel_name, (best_params, _) in best_params_all_kernels.items():
        start_time = time.time()
        svm_model = train_svm_with_libsvm(X_train, y_train, best_params)
        svm_training_time = time.time() - start_time
        svm_accuracy, svm_y_pred = evaluate_svm_with_libsvm(svm_model, X_test, y_test)
        plot_results(y_test, svm_y_pred, svm_accuracy, f"SVM_{kernel_name}")

        f.write(f"{kernel_name} Kernel SVM Accuracy: {svm_accuracy:.4f}, Training Time: {svm_training_time:.2f} seconds\n")
        f.write(f"{kernel_name} Kernel SVM Classification Report:\n{classification_report(y_test, svm_y_pred)}\n")

    knn_model = train_knn(X_train, y_train, n_neighbors=3)
    knn_accuracy, knn_y_pred = evaluate_model(knn_model, X_test, y_test, "k-NN")
    plot_results(y_test, knn_y_pred, knn_accuracy, "k-NN")

    f.write(f"k-NN Accuracy: {knn_accuracy:.4f}\n")
    f.write(f"k-NN Classification Report:\n{classification_report(y_test, knn_y_pred)}\n")

print(f"Results saved to {results_file}")
