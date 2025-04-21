# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Reshape y_train and y_test to 1D arrays
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Define classes
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Function to plot sample images
def plot_sample(X, y, index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

# Plot some sample images
plot_sample(X_train, y_train, 0)
plot_sample(X_train, y_train, 1)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build a simple artificial neural network (ANN)
ann = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the ANN
ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)

# Evaluate ANN
ann.evaluate(X_test, y_test)

# Print classification report for ANN
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report (ANN): \n", classification_report(y_test, y_pred_classes))

# Build a convolutional neural network (CNN)
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the CNN
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

# Evaluate CNN
cnn.evaluate(X_test, y_test)

# Plot a sample image and its predicted class
plot_sample(X_test, y_test, 3)
y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]
print("Predicted class:", classes[y_classes[3]])