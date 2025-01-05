import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import time
from load_data_cifer import load_cifar10_data  
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'cifar-10-batches-py'  
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

pca = PCA(0.90) 
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

class RBFNetwork:
    def __init__(self, num_centers, output_dim):
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.centers = None
        self.weights = None
        self.encoder = OneHotEncoder()

    def _rbf(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = self._rbf(x, c, self.sigma)
        return G

    def fit(self, X, y):
        y_one_hot = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        print("Starting K-Means clustering...")
        kmeans = KMeans(n_clusters=self.num_centers, random_state=0)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        print("K-Means clustering completed.")
        
        self.sigma = np.mean(euclidean_distances(self.centers))
        print(f"Calculated sigma value: {self.sigma:.2f}")
        
        print("Calculating interpolation matrix...")
        G = self._calculate_interpolation_matrix(X)
        print("Interpolation matrix calculation completed.")
        
        print("Computing weights...")
        self.weights = np.dot(np.linalg.pinv(G), y_one_hot)
        print("Weights computed successfully.")

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return np.argmax(predictions, axis=1)

num_centers = 100  
rbf_net = RBFNetwork(num_centers=num_centers, output_dim=10)

start_time = time.time()
rbf_net.fit(x_train_pca, y_train)
training_time = time.time() - start_time

start_time = time.time()
y_pred_rbf = rbf_net.predict(x_test_pca)
testing_time = time.time() - start_time

accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
classification_report_rbf = classification_report(y_test, y_pred_rbf)

print("RBF Network Results")
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"Accuracy: {accuracy_rbf * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix_rbf)
print("Classification Report:")
print(classification_report_rbf)

with open("results_report_all.txt", "w") as f:
    f.write("RBF Network Results\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Testing time: {testing_time:.2f} seconds\n")
    f.write(f"Accuracy: {accuracy_rbf * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix_rbf) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report_rbf + "\n\n")

    f.write("k-NN Classifier Results\n")
    k_values = [1, 3, 5]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        start_time = time.time()
        knn.fit(x_train_pca, y_train)
        training_time_knn = time.time() - start_time

        start_time = time.time()
        y_pred_knn = knn.predict(x_test_pca)
        testing_time_knn = time.time() - start_time

        accuracy_knn = accuracy_score(y_test, y_pred_knn)

        f.write(f"\nk = {k}\n")
        f.write(f"Training time: {training_time_knn:.2f} seconds\n")
        f.write(f"Testing time: {testing_time_knn:.2f} seconds\n")
        f.write(f"Accuracy: {accuracy_knn * 100:.2f}%\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred_knn)) + "\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred_knn) + "\n")

    centroid = NearestCentroid()
    start_time = time.time()
    centroid.fit(x_train_pca, y_train)
    training_time_centroid = time.time() - start_time

    start_time = time.time()
    y_pred_centroid = centroid.predict(x_test_pca)
    testing_time_centroid = time.time() - start_time

    accuracy_centroid = accuracy_score(y_test, y_pred_centroid)

    f.write("\nNearest Centroid Classifier Results\n")
    f.write(f"Training time: {training_time_centroid:.2f} seconds\n")
    f.write(f"Testing time: {testing_time_centroid:.2f} seconds\n")
    f.write(f"Accuracy: {accuracy_centroid * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_centroid)) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_centroid) + "\n")

print("Results have been written to 'results_report_all.txt'")

def plot_confusion_matrix(cm, title, labels, filename):
    """Plots and saves a confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename)
    plt.close()

labels = [str(i) for i in range(10)]  
plot_confusion_matrix(conf_matrix_rbf, "RBF Network Confusion Matrix", labels, "conf_matrix_rbf.png")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_pca, y_train)
    y_pred_knn = knn.predict(x_test_pca)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plot_confusion_matrix(cm_knn, f"k-NN Confusion Matrix (k={k})", labels, f"conf_matrix_knn_k{k}.png")

cm_centroid = confusion_matrix(y_test, y_pred_centroid)
plot_confusion_matrix(cm_centroid, "Nearest Centroid Confusion Matrix", labels, "conf_matrix_centroid.png")

print("Confusion matrix plots have been saved as PNG files.")
