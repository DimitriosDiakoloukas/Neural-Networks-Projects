import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from load_data_cifer import load_cifar10_data  

class RBFNetwork:
    def __init__(self, num_centers, output_dim, regularization=1e-6):
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.centers = None
        self.sigmas = None
        self.weights = None
        self.encoder = OneHotEncoder()
        self.regularization = regularization
        self.losses = [] 

    def _rbf_kernel(self, X, centers, sigmas):
        n_samples, n_features = X.shape
        n_centers = centers.shape[0]
        kernel_matrix = np.zeros((n_samples, n_centers), dtype=np.float32)

        batch_size = 1000  
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]  
            distances = np.linalg.norm(batch[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            kernel_matrix[i:i+batch_size, :] = np.exp(-distances**2 / (2 * sigmas**2))

        return kernel_matrix

    def _compute_sigmas(self, centers):
        distances = euclidean_distances(centers)
        k = max(1, int(0.1 * self.num_centers)) 
        sigmas = np.mean(np.sort(distances, axis=1)[:, 1:k+1], axis=1)  
        return sigmas

    def fit(self, X, y):
        y_one_hot = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()

        print("Fitting K-Means clustering to find RBF centers...")
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        print("RBF centers determined.")

        print("Calculating sigmas for RBF centers...")
        self.sigmas = self._compute_sigmas(self.centers)
        print("Sigmas calculated.")

        print("Computing RBF kernel matrix...")
        G = self._rbf_kernel(X, self.centers, self.sigmas)
        print("RBF kernel matrix computed.")

        print("Calculating weights with regularization...")
        regularization_matrix = self.regularization * np.eye(G.shape[1])
        self.weights = np.linalg.solve(G.T @ G + regularization_matrix, G.T @ y_one_hot)
        print("Weights computed.")

        print("Tracking training loss...")
        for i in range(X.shape[0]):
            batch = X[i:i + 1]  
            G_batch = self._rbf_kernel(batch, self.centers, self.sigmas)
            y_pred = G_batch @ self.weights
            loss = np.mean((y_one_hot[i:i + 1] - y_pred) ** 2)
            self.losses.append(loss)

    def predict(self, X):
        G = self._rbf_kernel(X, self.centers, self.sigmas)
        predictions = G @ self.weights
        return np.argmax(predictions, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, cm, report


if __name__ == "__main__":
    print("Loading CIFAR-10 dataset...")
    data_dir = 'cifar-10-batches-py' 
    (X_train, y_train), (X_test, y_test) = load_cifar10_data(data_dir)

    print("Preprocessing CIFAR-10 dataset...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_flat = X_train_flat.astype('float32') / 255.0
    X_test_flat = X_test_flat.astype('float32') / 255.0

    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=300) 
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    print("Initializing RBF Neural Network...")
    num_centers = 500  
    output_dim = 10   
    rbf_net = RBFNetwork(num_centers=num_centers, output_dim=output_dim, regularization=1e-5)

    print("Training RBF Neural Network...")
    rbf_net.fit(X_train_pca, y_train)

    print("Evaluating RBF Neural Network...")
    accuracy, cm, report = rbf_net.evaluate(X_test_pca, y_test)

    results_text = (
        f"\nAccuracy: {accuracy * 100:.2f}%\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Classification Report:\n{report}\n"
    )

    with open('improved_rbf_results.txt', 'w') as f:
        f.write(results_text)

    print("Results saved to 'rbf_results.txt'.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('improved_confusion_matrix.png')

    print("Confusion matrix saved as 'improved_confusion_matrix.png'.")

    plt.figure(figsize=(8, 6))
    plt.plot(rbf_net.losses, label='Training Loss')
    plt.title('Training Loss over Time')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('improved_training_loss.png')

    print("Training loss plot saved as 'improved_training_loss.png'.")
