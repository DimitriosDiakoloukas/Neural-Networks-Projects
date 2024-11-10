import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from load_data_cifer import load_cifar10_data

data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

num_classes = 10
batch_size = 128
epochs = 100

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization()) 
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=10, height_shift_range=0.1, width_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
traingen = datagen.flow(x_train, y_train, batch_size=batch_size)
history = model.fit(traingen, epochs=epochs, validation_data=(x_test, y_test))

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("training_history_CNN_Augmented.png")
plt.close()  

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
y_pred = model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)
true_labels = np.argmax(y_test, axis=1)
wrong_indices = np.where(predicted_labels != true_labels)[0]
correct_indices = np.where(predicted_labels == true_labels)[0]
    
fig, axes = plt.subplots(2, 7, figsize=(15, 6))
fig.suptitle('Correct and Incorrect Predictions', fontsize=16)

for i in range(7):
    idx=correct_indices[i] 
    ax=axes[0, i]
    ax.imshow(x_test[idx])
    pred_label=np.argmax(y_pred[idx]) 
    true_label=np.argmax(y_test[idx])
    ax.set_title(f"Pred: {pred_label}, True: {true_label}")
    ax.axis('off')

for i in range(7):
    idx=wrong_indices[i]  
    ax=axes[1, i]
    ax.imshow(x_test[idx])
    pred_label=np.argmax(y_pred[idx]) 
    true_label=np.argmax(y_test[idx])
    ax.set_title(f"Pred: {pred_label}, True: {true_label}")
    ax.axis('off')

plt.savefig("sample_predictions_CNN_Augmented.png")  
plt.close() 
print()
print(f"Accuracy CNN with data augmentations: {test_acc}")