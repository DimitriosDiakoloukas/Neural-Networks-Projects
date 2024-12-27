# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten, MaxPooling2D
# from tensorflow.keras.models import Sequential
# import tensorflow.keras.backend as K
# import numpy as np
# import matplotlib.pyplot as plt
# from load_data_cifer import load_cifar10_data

# class RBFNNLayer(Layer):
#     def __init__(self, num_centers, output_dim, **kwargs):
#         super(RBFNNLayer, self).__init__(**kwargs)
#         self.num_centers = num_centers
#         self.output_dim = output_dim

#     def build(self, input_shape):
#         self.centers = self.add_weight(
#             name="centers",
#             shape=(self.num_centers, input_shape[-1]),
#             initializer="random_uniform",
#             trainable=True
#         )

#         self.betas = self.add_weight(
#             name="betas",
#             shape=(self.num_centers,),
#             initializer="ones",
#             trainable=True
#         )

#         self.linear_weights = self.add_weight(
#             name="linear_weights",
#             shape=(self.num_centers, self.output_dim),
#             initializer="random_normal",
#             trainable=True
#         )

#         self.biases = self.add_weight(
#             name="biases",
#             shape=(self.output_dim,),
#             initializer="zeros",
#             trainable=True
#         )

#         super(RBFNNLayer, self).build(input_shape)

#     def call(self, inputs):
#         expanded_inputs = K.expand_dims(inputs, axis=1)
#         distances = K.sum(K.square(expanded_inputs - self.centers), axis=2)
#         rbf_activations = K.exp(-self.betas * distances)
#         output = K.dot(rbf_activations, self.linear_weights) + self.biases

#         return output

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

# def create_rbfnn_model(input_dim, num_centers, output_dim):
#     model = Sequential()
#     model.add(tf.keras.Input(shape=(input_dim,)))
#     model.add(RBFNNLayer(num_centers=num_centers, output_dim=output_dim))
#     model.add(Dense(output_dim, activation="softmax"))
#     return model

# data_dir = 'cifar-10-batches-py'
# (X_train, y_train), (X_test, y_test) = load_cifar10_data(data_dir)

# num_classes = 10
# X_train = X_train.reshape(-1, 32 * 32 * 3)
# X_test = X_test.reshape(-1, 32 * 32 * 3)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# input_dim = 32 * 32 * 3
# num_centers = 100
# output_dim = num_classes

# conv_model = Sequential([
#     tf.keras.layers.Reshape((32, 32, 3), input_shape=(input_dim,)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])

# conv_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# conv_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# X_train_features = conv_model.predict(X_train)
# X_test_features = conv_model.predict(X_test)

# rbfnn_model = create_rbfnn_model(num_classes, num_centers, output_dim)
# rbfnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# history = rbfnn_model.fit(X_train_features, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# loss, accuracy = rbfnn_model.evaluate(X_test_features, y_test, verbose=0)
# print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# plt.figure()
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from load_data_cifer import load_cifar10_data
import time

class RBFNNLayer(Layer):
    def __init__(self, num_centers, output_dim, **kwargs):
        super(RBFNNLayer, self).__init__(**kwargs)
        self.num_centers = num_centers
        self.output_dim = output_dim

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=(self.num_centers, input_shape[-1]),
            initializer="random_uniform",
            trainable=True
        )
        self.betas = self.add_weight(
            name="betas",
            shape=(self.num_centers,),
            initializer="ones",
            trainable=True
        )
        self.linear_weights = self.add_weight(
            name="linear_weights",
            shape=(self.num_centers, self.output_dim),
            initializer="random_normal",
            trainable=True
        )
        self.biases = self.add_weight(
            name="biases",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True
        )
        super(RBFNNLayer, self).build(input_shape)

    def call(self, inputs):
        expanded_inputs = K.expand_dims(inputs, axis=1)
        distances = K.sum(K.square(expanded_inputs - self.centers), axis=2)
        rbf_activations = K.exp(-self.betas * distances)
        output = K.dot(rbf_activations, self.linear_weights) + self.biases
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def create_rbfnn_model(input_dim, num_centers, output_dim):
    inputs = Input(shape=(input_dim,))
    x = RBFNNLayer(num_centers=num_centers, output_dim=output_dim)(inputs)
    outputs = Dense(output_dim, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_rbfnn_with_parameters(num_centers, epochs):
    rbfnn_model = create_rbfnn_model(X_train_features.shape[1], num_centers, output_dim)
    rbfnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = rbfnn_model.fit(X_train_features, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=1)
    loss, accuracy = rbfnn_model.evaluate(X_test_features, y_test, verbose=0)
    return history, loss, accuracy

data_dir = 'cifar-10-batches-py'
(X_train, y_train), (X_test, y_test) = load_cifar10_data(data_dir)

num_classes = 10
X_train = X_train.reshape(-1, 32 * 32 * 3).astype("float32") / 255.0
X_test = X_test.reshape(-1, 32 * 32 * 3).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

input_dim = 32 * 32 * 3
output_dim = num_classes

cnn_inputs = Input(shape=(input_dim,))
x = tf.keras.layers.Reshape((32, 32, 3))(cnn_inputs)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
features = Dense(100, activation='relu', name='features')(x)  
outputs = Dense(num_classes, activation='softmax')(features)
conv_model = Model(inputs=cnn_inputs, outputs=outputs)

conv_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
conv_model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)

feature_extractor = Model(inputs=conv_model.input, outputs=conv_model.get_layer('features').output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

results = []
for centers in [50, 100, 200]:
    history, loss, accuracy = evaluate_rbfnn_with_parameters(num_centers=centers, epochs=10)
    results.append((centers, history, loss, accuracy))

for centers, history, loss, accuracy in results:
    print(f"Centers: {centers}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

last_history = results[-1][1]
plt.figure()
plt.plot(last_history.history['loss'], label='Training Loss')
plt.plot(last_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')

plt.figure()
plt.plot(last_history.history['accuracy'], label='Training Accuracy')
plt.plot(last_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')
