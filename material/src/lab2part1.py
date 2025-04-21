import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Print the first few labels and corresponding images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xlabel("Class: " + str(y_train[i]))
plt.show()

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

X_train = X_train / 255.0
X_test = X_test / 255.0

ann = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(300, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)

ann.evaluate(X_test, y_test)

y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report (ANN): \n", classification_report(y_test, y_pred_classes))

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=3)

cnn.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)

def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index], cmap='gray')
    plt.xlabel("Digit: " + str(y[index]))

plot_sample(X_test, y_test, 3)

y_pred = cnn.predict(X_test.reshape(-1, 28, 28, 1))
y_classes = [np.argmax(element) for element in y_pred]
print("Predicted class:", y_classes[3])
