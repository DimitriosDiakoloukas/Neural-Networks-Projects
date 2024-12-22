import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from libsvm.svmutil import *
import time
from load_data_cifer import load_cifar10_data

# Perform PCA for dimensionality reduction
def perform_pca_specific_variance(data, n_components):
    print("Performing PCA with specific variance...")
    pca = PCA(n_components)
    transformed_data = pca.fit_transform(data)
    print(f"Number of principal components: {pca.n_components_}")
    return transformed_data

# Train SVM using libsvm
def train_svm_with_libsvm(x_train, y_train, params):
    print("Training SVM with libsvm...")
    svm_model = svm_train(y_train.tolist(), x_train.tolist(), params)
    return svm_model

# Evaluate SVM using libsvm
def evaluate_svm_with_libsvm(svm_model, x_test, y_test):
    print("Evaluating SVM...")
    y_pred, _, _ = svm_predict(y_test.tolist(), x_test.tolist(), svm_model)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Perform grid search for RBF kernel
def grid_search_rbf_kernel(X_train, y_train):
    print("Performing grid search with cross-validation for RBF kernel...")
    best_params = None
    best_accuracy = 0
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.0001, 0.001, 0.01, 0.1]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for C in C_values:
        for gamma in gamma_values:
            params = f"-t 2 -c {C} -g {gamma}"  # RBF kernel: -t 2
            fold_accuracies = []

            for train_idx, val_idx in kf.split(X_train):
                x_fold_train, x_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                svm_model = train_svm_with_libsvm(x_fold_train, y_fold_train, params)
                accuracy, _ = evaluate_svm_with_libsvm(svm_model, x_fold_val, y_fold_val)
                fold_accuracies.append(accuracy)

            mean_accuracy = np.mean(fold_accuracies)
            print(f"Params: C={C}, gamma={gamma}, CV Accuracy={mean_accuracy:.4f}")

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = params

    print(f"Best Params for RBF kernel: {best_params}, Best CV Accuracy: {best_accuracy:.4f}")
    return best_params

# Plot results
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
    plot_file = f"FINAL_confusion_matrix_{model_name}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Confusion matrix saved as {plot_file}")

# Load CIFAR-10 dataset
data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

# Preprocess the dataset
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test / 255.0

num_classes = 10
y_train = np.array(y_train)
y_test = np.array(y_test)

# Flatten the data
X = np.vstack((x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
y = np.hstack((y_train, y_test))

# Perform PCA for dimensionality reduction
pca_variance = 0.85  # Adjusted to retain 85% variance
X_pca = perform_pca_specific_variance(X, pca_variance)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Grid search for RBF kernel
start_time = time.time()
best_rbf_params = grid_search_rbf_kernel(X_train, y_train)
grid_search_time = time.time() - start_time
print(f"Grid search completed in {grid_search_time:.2f} seconds.")

# Train final RBF model
start_time = time.time()
final_rbf_model = train_svm_with_libsvm(X_train, y_train, best_rbf_params)
training_time = time.time() - start_time
print(f"Final RBF model trained in {training_time:.2f} seconds.")

# Evaluate the final model
start_time = time.time()
final_accuracy, y_pred = evaluate_svm_with_libsvm(final_rbf_model, X_test, y_test)
evaluation_time = time.time() - start_time
print(f"Evaluation completed in {evaluation_time:.2f} seconds.")
print(f"Final RBF Kernel Accuracy: {final_accuracy:.4f}")

# Save results and plot confusion matrix
results_file = "svm_rbf_results.txt"
with open(results_file, "w") as f:
    f.write(f"Final RBF Kernel SVM Accuracy: {final_accuracy:.4f}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n")
    f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
plot_results(y_test, y_pred, final_accuracy, "RBF")

print(f"Results saved to {results_file}")
