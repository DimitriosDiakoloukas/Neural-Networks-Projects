import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import time
from load_data_cifer import load_cifar10_data

def perform_pca(data, variance_ratio):
    print("Performing PCA...")
    pca = PCA(variance_ratio)
    reduced_data = pca.fit_transform(data)
    print(f"Number of components selected: {pca.n_components_}")
    return reduced_data, pca.n_components_

def train_svm(features, labels, hyperparams=None):
    if hyperparams is None:
        hyperparams = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
    svm_model = OneVsOneClassifier(SVC(**hyperparams))
    svm_model.fit(features, labels)
    return svm_model

def evaluate_svm(model, features, labels):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    return accuracy, predictions

if __name__ == "__main__":
    results_file = "svc_results_with_pca.txt"
    
    data_dir = 'cifar-10-batches-py'
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

    with open(results_file, "w") as f:
        f.write(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}\n")
        f.write(f"Test data shape: {x_test.shape}, Labels: {y_test.shape}\n")

        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)

        variance_to_retain = 0.90
        x_train_pca, num_components_train = perform_pca(x_train_flat, variance_to_retain)
        x_test_pca, num_components_test = perform_pca(x_test_flat, variance_to_retain)

        f.write(f"Number of PCA components retained: {num_components_train}\n")

        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            x_train_pca, y_train, test_size=0.2, random_state=42
        )

        reduced_param_dist = {
            'C': [0.1, 1],  
            'gamma': ['scale', 'auto'],  
            'kernel': ['linear', 'rbf']  
        }

        svm_search = RandomizedSearchCV(
            SVC(),
            param_distributions=reduced_param_dist,
            cv=2,  
            n_iter=5,  
            random_state=42,
            n_jobs=-1
        )

        f.write("Starting hyperparameter search...\n")
        start_search_time = time.time()
        svm_search.fit(x_train_split, y_train_split)
        search_time = time.time() - start_search_time

        best_params = svm_search.best_params_
        f.write(f"Best parameters found: {best_params}\n")
        f.write(f"Hyperparameter search time: {search_time:.2f} seconds\n")

        final_svm = train_svm(x_train_pca, y_train, best_params)

        start_eval_time = time.time()
        test_accuracy, predictions = evaluate_svm(final_svm, x_test_pca, y_test)
        evaluation_time = time.time() - start_eval_time

        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Evaluation time: {evaluation_time:.2f} seconds\n")

        f.write(f"Predictions: {predictions[:10]}... (first 10 predictions)\n")

    print(f"Results saved to {results_file}")
