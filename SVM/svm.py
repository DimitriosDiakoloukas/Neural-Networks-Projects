import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import RMSprop
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier   
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import time

#cifar given unpickle function
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#train_batches array to have all the data from the cifar batches
def load_all_batches(path):
    train_batches = []
    for i in range(1, 6):
        file_path = f'{path}/data_batch_{i}'
        train_batches.append(unpickle(file_path))
    return train_batches

#loading test_batch for test set
def load_test_batch(path):
    file_path = f'{path}/test_batch'
    return unpickle(file_path)

#combine the data and the labels of all the cifar batches to single variables
def combine(batches):
    combined_data = np.concatenate([batch[b'data'] for batch in batches], axis=0)
    combined_labels = np.concatenate([batch[b'labels'] for batch in batches], axis=0)
    return combined_data, combined_labels

#split data (x_train. y_train, x_test, y_test) 
def split_data(path, test_size=0.2, random_state=42):
    train_batches = load_all_batches(path)
    test_batch = load_test_batch(path)
    
    x_train, y_train = combine(train_batches)
    x_test, y_test = test_batch[b'data'], np.array(test_batch[b'labels'])
    
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)
    x_train = x_train/255.0
    
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test

#we perform pca given a desired variance percentage so the module automatically decides the dimensions
def perform_pca_specific_variance(data, n_components):
    print("Entered pca function")
    
    pca = PCA(n_components)
    
    transformed_data = pca.fit_transform(data)
    
    print("Number of principal components: ", pca.n_components_)
    print("Amount of variance explained(captured) by each component: ", pca.explained_variance_)
    
    return transformed_data

def train_svm(x_train, y_train, params):
    if params is None:
        params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}

    svm = OneVsOneClassifier(SVC(**params))
    svm.fit(x_train, y_train)
    return svm

def evaluate_svm(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

x_train, y_train, x_test, y_test = split_data('cifar-10-batches-py', test_size=0.2, random_state=42)

print('Training Data Shape: ', x_train.shape)
print('Testing Data Shape: ', x_test.shape)

print('Label Training Data Shape: ', y_train.shape)
print('Label Testing Data Shape: ', y_test.shape)
    
classes = np.unique(y_train)
print("number of classes: ", classes)

transform_train = perform_pca_specific_variance(x_train, 2)
transform_test = perform_pca_specific_variance(x_test, 2)

param_dist = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

svm_estimator = SVC()

random_search = RandomizedSearchCV(
        svm_estimator,
        param_distributions=param_dist,
        cv=2,
        random_state=42,
        n_jobs=-1  # Use parallel processing if available
    )

print("Starting randoom search")
begin = time.time()

#random_search.fit(transform_train, y_train)

end_of_search = time.time()

print("TIme for search: ", end_of_search - begin)

svm_model = train_svm(transform_train, y_train, None)

end_of_model_train = time.time()

print("Model trained at time point: ", end_of_model_train)

accuracy = evaluate_svm(svm_model, transform_test, y_test)
