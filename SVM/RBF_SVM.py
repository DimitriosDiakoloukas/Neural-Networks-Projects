import matplotlib.pyplot as plt
from itertools import product
from libsvm.svmutil import svm_train, svm_predict
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import random
import time
from load_data_cifer import load_cifar10_data
import numpy as np


def reduce_dataset(images, labels, max_samples):
    indices = list(range(len(images)))
    random.shuffle(indices)
    selected_indices = indices[:max_samples]
    reduced_images = [images[i] for i in selected_indices]
    reduced_labels = [labels[i] for i in selected_indices]
    return reduced_images, reduced_labels

def apply_pca(train_features, test_features, n_components=100):
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    print(f"\nApplying PCA to reduce dimensionality to {n_components} components...")
    pca = PCA(n_components=n_components)
    train_features_reduced = pca.fit_transform(train_features)
    test_features_reduced = pca.transform(test_features)
    print(f"Reduced dimensionality: {train_features.shape[1]} -> {n_components}")
    return train_features_reduced.tolist(), test_features_reduced.tolist()

def grid_search_rbf(alpha, beta, gamma_weight, train_labels, train_features, test_labels, test_features, C_values, gamma_values):
    print("\nPerforming Grid Search for RBF Kernel...\n")
    best_score = 0
    best_params = {}
    grid_results = {}

    for C, gamma in product(C_values, gamma_values):
        param_str = f"-s 0 -t 2 -c {C} -g {gamma} -q"
        print(f"Testing parameters: C={C}, gamma={gamma}, kernel=rbf")

        start_time = time.time()
        temp_model = svm_train(train_labels, train_features, param_str)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

        accuracy_train = svm_predict(train_labels, train_features, temp_model)
        num_support_vectors, accuracy, predicted_labels, _ = evaluate(temp_model, test_labels, test_features)
        grid_results[(C, gamma)] = accuracy[0]

        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        print(f"F1 Score: {f1:.4f}")

        support_vector_penalty = gamma_weight * (num_support_vectors / len(train_labels))

        combined_score = alpha * accuracy_train[1][0] + beta * f1 - support_vector_penalty

        print(f"Number of Support Vectors: {num_support_vectors}")
        print(f"Training Set Size: {len(train_labels)}")
        print(f"Percentage of Support Vectors: {100 * num_support_vectors / len(train_labels):.2f}%")
        print("\n")

        if combined_score > best_score:
            best_score = combined_score
            best_params = {'C': C, 'gamma': gamma}

    print("\nGrid Search Complete")
    print(f"Best Parameters: {best_params}")
    return best_params, grid_results


def evaluate(model, test_labels, test_features):
    print("\nEvaluating model on the test set...")
    predicted_labels, accuracy, pred_values = svm_predict(test_labels, test_features, model)
    num_support_vectors = model.get_nr_sv()
    return num_support_vectors, accuracy, predicted_labels, pred_values


def plot_results_rbf(C_values, gamma_values, grid_results, name_prefix="grid_search_rbf"):
    plt.figure(figsize=(8, 6))
    plt.title("Grid Search Accuracy for RBF Kernel", fontsize=16)
    plt.xlabel("Gamma", fontsize=14)
    plt.ylabel("C", fontsize=14)

    accuracy_matrix = [[grid_results[(C, gamma)] for gamma in gamma_values] for C in C_values]
    plt.imshow(accuracy_matrix, interpolation='nearest', cmap='viridis', origin='lower', aspect='auto',
               extent=[min(gamma_values), max(gamma_values), min(C_values), max(C_values)])
    plt.colorbar(label="Accuracy (%)")
    plt.xticks(gamma_values, [f"{g:.2e}" for g in gamma_values], fontsize=12)
    plt.yticks(C_values, [f"{c}" for c in C_values], fontsize=12)
    plt.savefig(f"{name_prefix}.png")
    print(f"Plot saved as '{name_prefix}.png'")


def retrain_with_best_params(best_params, train_labels, train_features):
    C = best_params['C']
    gamma = best_params['gamma']
    param_str = f"-s 0 -t 2 -c {C} -g {gamma} -q"
    print(f"\nRetraining SVM with best parameters: C={C}, gamma={gamma} (RBF Kernel)...")
    final_model = svm_train(train_labels, train_features, param_str)
    return final_model


def write_results_to_file(best_params, grid_results, final_accuracy, num_support_vectors, filename="rbf.txt"):
    with open(filename, "w") as f:
        f.write("Grid Search Results for RBF Kernel\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write("\nGrid Search Results:\n")
        for (C, gamma), accuracy in grid_results.items():
            f.write(f"C={C}, gamma={gamma}, accuracy={accuracy:.2f}%\n")
        f.write("\nFinal Model Evaluation:\n")
        f.write(f"Accuracy: {final_accuracy[0]:.2f}%\n")
        f.write(f"Number of Support Vectors: {num_support_vectors}\n")
    print(f"Results written to {filename}")


def main():
    data_directory = 'cifar-10-batches-py'
    (x_train_data, y_train_data), (x_test_data, y_test_data) = load_cifar10_data(data_directory)

    class_filter = [0, 1]
    train_indices = [i for i, label in enumerate(y_train_data) if label in class_filter]
    test_indices = [i for i, label in enumerate(y_test_data) if label in class_filter]

    img_train = [x_train_data[i].flatten() for i in train_indices]
    label_train = [y_train_data[i] for i in train_indices]
    img_test = [x_test_data[i].flatten() for i in test_indices]
    label_test = [y_test_data[i] for i in test_indices]

    max_samples = 60000
    img_train, label_train = reduce_dataset(img_train, label_train, max_samples)

    n_components = 100
    img_train_reduced, img_test_reduced = apply_pca(img_train, img_test, n_components)

    alpha = 0.40
    beta = 0.60
    gamma_weight = 10
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

    best_params, grid_results = grid_search_rbf(
        alpha, beta, gamma_weight,
        label_train, img_train_reduced,
        label_test, img_test_reduced,
        C_values, gamma_values
    )

    final_model = retrain_with_best_params(best_params, label_train, img_train_reduced)

    num_support_vectors, final_accuracy, predicted_labels, _ = evaluate(final_model, label_test, img_test_reduced)
    print(f"\nFinal Model Evaluation:")
    print(f"Accuracy: {final_accuracy[0]:.2f}%")
    print(f"Number of Support Vectors: {num_support_vectors}")

    write_results_to_file(best_params, grid_results, final_accuracy, num_support_vectors, filename="rbf.txt")

    name = "grid_search_rbf_2_classes_with_pca_60000"
    plot_results_rbf(C_values, gamma_values, grid_results, name)


if __name__ == "__main__":
    main()
