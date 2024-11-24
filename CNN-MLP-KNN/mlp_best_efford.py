import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical
from load_data_cifer import load_cifar10_data

data_dir = 'cifar-10-batches-py'  
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

num_classes=10
epochs=30
batch_size=60

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))  
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

plt.figure(figsize=(15, 6))
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

plt.savefig("training_history_MLP_BEST.png")
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

plt.savefig("sample_predictions_MLP_BEST.png")  
plt.close() 

print()
print(f"MLP model accuracy: {test_acc}")