import numpy as np
from collections import Counter
from load_data_cifer import load_cifar10_data

def preprocess():
    data_dir = 'cifar-10-batches-py'
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)    
    x_train_1D = x_train.reshape(x_train.shape[0], -1)
    x_test_1D = x_test.reshape(x_test.shape[0], -1)
    x_train_subset, y_train_subset = x_train_1D[:2000], y_train[:2000]
    x_test_subset, y_test_subset = x_test_1D[:2000], y_test[:2000]
    return x_train_subset, y_train_subset, x_test_subset, y_test_subset

def knn_predict(x_train, y_train, x_test, k):
    y_pred = []
    for idx, x in enumerate(x_test):
        distances = np.linalg.norm(x_train - x, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        #print(f"\ntest point {idx+1}:")
        #print("Distances to all points:", distances[:10])  
        #print("Indices of nearest neighbors:", nearest_indices)
        #print("Labels of nearest neighbors:", nearest_labels)
        common = Counter(nearest_labels).most_common(1)[0][0]
        y_pred.append(common)        
        #print("Predicted Label:", common)
    return np.array(y_pred)

def nearest_centroid(x_train, y_train, x_test):
    centroids = {}
    for label in np.unique(y_train): 
        class_points = [] 
        for i in range(len(y_train)):
            if y_train[i] == label:  
                class_points.append(x_train[i])  
        class_points = np.array(class_points)
        centroids[label] = class_points.mean(axis=0)
    predictions = []
    for test_point in x_test:  
        distances = {}
        for label, centroid in centroids.items():
            distance = np.linalg.norm(test_point - centroid) 
            distances[label] = distance
        closest_label = min(distances, key=distances.get)
        predictions.append(closest_label)
    return np.array(predictions)

def eval(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: 
            correct += 1  
    accuracy = correct / len(y_true)
    return accuracy

x_train, y_train, x_test, y_test = preprocess()

y_pred_knn_1 = knn_predict(x_train, y_train, x_test, k=1)
accuracy_knn_1 = eval(y_test, y_pred_knn_1)
print(f"KNN k=1 accuracy: {accuracy_knn_1}")

y_pred_knn_3 = knn_predict(x_train, y_train, x_test, k=3)
accuracy_knn_3 = eval(y_test, y_pred_knn_3)
print(f"KNN k=3 accuracy: {accuracy_knn_3}")

y_pred_centroid = nearest_centroid(x_train, y_train, x_test)
accuracy_centroid = eval(y_test, y_pred_centroid)
print(f"Nearest centroid accuracy: {accuracy_centroid}")
