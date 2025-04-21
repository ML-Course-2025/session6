# Activity 2: 

---
## Part 1/3: Introduction to Image Classification with CNNs

### Objective:

By the end of this activity, you will gain practical experience in building and training CNN models for image classification tasks using the MNIST dataset. Through this lab, you will achieve the following objectives:

1. Understand the basics of image classification and Convolutional Neural Networks (CNNs).
2. Learn how to preprocess image data for training.
3. Build and train a simple Artificial Neural Network (ANN) model for image classification.
4. Construct and train a Convolutional Neural Network (CNN) model for improved performance.
5. Evaluate model performance and visualize predictions.

### Lab Steps:

**Step 1: Import Libraries**

- Before we dive into coding, let's import the necessary libraries. We'll need TensorFlow and its Keras API for building our models, Matplotlib for visualization, NumPy for numerical operations, and Scikit-Learn for performance evaluation metrics.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
```

**Step 2: Load and Explore Dataset**

- Now, let's load the MNIST dataset and take a closer look at its structure. We'll examine the shapes of the training and test data arrays and peek into a few sample images with their corresponding labels.

```python
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Print the first few labels and corresponding images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xlabel("Label: " + str(y_train[i]))
plt.show()
```

**Step 3: Preprocess Data**

- Before feeding the data into our models, we need to preprocess it. This involves reshaping the labels into 1D arrays and normalizing the image data to ensure consistent scaling between 0 and 1.

```python
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

X_train = X_train / 255.0
X_test = X_test / 255.0
```

**Step 4: Build and Train a Simple ANN Model**

- Let's start with a simple Artificial Neural Network (ANN) model. We'll define the architecture, compile it with suitable parameters, and train it on our preprocessed data for a few epochs.

```python
ann = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)
```

**Step 5: Evaluate the ANN Model**

- Time to assess the performance of our ANN model on the test data. We'll evaluate its accuracy and print a classification report to analyze its performance across different classes.

```python
ann.evaluate(X_test, y_test)

y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report (ANN): \n", classification_report(y_test, y_pred_classes))
```

**Step 6: Build and Train a CNN Model**

- Now, let's level up by building a Convolutional Neural Network (CNN) model. We'll define the architecture with convolutional and pooling layers, compile it, and train it on our dataset.

```python
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=3)
```

**Step 7: Evaluate the CNN Model**

- With our CNN model trained, it's crucial to evaluate its performance. We'll assess its accuracy on the test data to ensure it's performing as expected.

```python
cnn.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
```

**Step 8: Visualize Predictions**

- Visualizing predictions can offer deeper insights into our model's performance. We'll plot a sample image along with its predicted class label to understand how our model is making predictions.

```python
def plot_sample(X, y, index):
    plt.figure(figsize=(2, 2))
    plt.imshow(X[index], cmap='gray')
    plt.title('Label: ' + str(y[index]))

plot_sample(X_test, y_test,

 0)

y_pred = cnn.predict(X_test.reshape(-1, 28, 28, 1))
y_classes = [np.argmax(element) for element in y_pred]
print("Predicted class:", y_classes[0])
```

**Step 9**

> Here's the complete [source code](./src/lab2part1.py)


------------
## Part 2/3: Introduction to Image Classification with CNNs

### Objective:

By the end of this activity, you will gain practical experience in building and training CNN models for image classification tasks using the CIFAR-10 dataset. Through this lab, you will achieve the following objectives:

1. Understand the basics of image classification and Convolutional Neural Networks (CNNs).
2. Learn how to preprocess image data for training.
3. Build and train a simple Artificial Neural Network (ANN) model for image classification.
4. Construct and train a Convolutional Neural Network (CNN) model for improved performance.
5. Evaluate model performance and visualize predictions.

### Steps:

**Step 1: Import Libraries**

- Before we dive into coding, let's import the necessary libraries. We'll need TensorFlow and its Keras API for building our models, Matplotlib for visualization, NumPy for numerical operations, and Scikit-Learn for performance evaluation metrics.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
```

**Step 2: Load and Explore Dataset**

- Now, let's load the CIFAR-10 dataset and take a closer look at its structure. We'll examine the shapes of the training and test data arrays and peek into a few sample images with their corresponding labels.

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Print the first few labels and corresponding images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.xlabel("Class: " + str(y_train[i]))
plt.show()
```

**Step 3: Preprocess Data**

- Before feeding the data into our models, we need to preprocess it. This involves reshaping the labels into 1D arrays and normalizing the image data to ensure consistent scaling between 0 and 1.

```python
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

X_train = X_train / 255.0
X_test = X_test / 255.0
```

**Step 4: Build and Train a Simple ANN Model**

- Let's start with a simple Artificial Neural Network (ANN) model. We'll define the architecture, compile it with suitable parameters, and train it on our preprocessed data for a few epochs.

```python
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)
```

**Step 5: Evaluate the ANN Model**

- Time to assess the performance of our ANN model on the test data. We'll evaluate its accuracy and print a classification report to analyze its performance across different classes.

```python
ann.evaluate(X_test, y_test)

y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print("Classification Report (ANN): \n", classification_report(y_test, y_pred_classes))
```

**Step 6: Build and Train a CNN Model**

- Now, let's level up by building a Convolutional Neural Network (CNN) model. We'll define the architecture with convolutional and pooling layers, compile it, and train it on our dataset.

```python
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
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

cnn.fit(X_train, y_train, epochs=10)
```

**Step 7: Evaluate the CNN Model**

- With our CNN model trained, it's crucial to evaluate its performance. We'll assess its accuracy on the test data to ensure it's performing as expected.

```python
cnn.evaluate(X_test, y_test)
```

**Step 8: Visualize Predictions**

- Visualizing predictions can offer deeper insights into our model's performance. We'll plot a sample image along with its predicted class label to understand how our model is making predictions.

```python
classes = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]

def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

plot_sample(X_test, y_test, 3)

y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]
print("Predicted class:", classes[y_classes[3]])
```

**Step 9**

> Here's the complete [source code](./src/lab2part2.py)


---
## Part 3/3: Discussion

- Refer to this code snippet:
```python
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
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

cnn.fit(X_train, y_train, epochs=10)
```

1. What type of neural network architecture is being defined in this code snippet?
2. What role does the `Sequential()` function play in this code?
3. How many convolutional layers are included in this convolutional neural network (CNN)?
4. What are the purposes of the `Conv2D` layers in this CNN?
5. What does the `(3, 3)` parameter represent in the `kernel_size` argument of the `Conv2D` layers?
6. What activation function is used for the convolutional layers, and why might this be a common choice for CNNs?
7. How does the `MaxPooling2D` layer contribute to the CNN architecture?
8. What does the `Flatten()` layer do in the CNN?
9. How many neurons are there in the first `Dense` layer, and why might this number have been chosen?
10. What activation function is used in the last `Dense` layer, and why is it suitable for this classification task?
11. What loss function and optimizer are specified in the compilation of the CNN, and why might they be appropriate for this task?
12. What metrics are being monitored during training, and why are they relevant for evaluating the model's performance?
13. How many epochs are specified for training the CNN, and what does this parameter represent?
14. What datasets (`X_train` and `y_train`) are being used for training the CNN?
15. What is the expected shape of the input data (`X_train`) for this CNN?

<details>
<summary>Sample answers</summary>

1. The code snippet defines a Convolutional Neural Network (CNN) architecture.

2. The `Sequential()` function is used to create a sequential model, allowing layers to be added in a sequential manner.

3. There are two convolutional layers included in this CNN.

4. The `Conv2D` layers perform convolution operations, extracting features from the input images.

5. The `(3, 3)` parameter represents the size of the convolutional kernels, which are 3x3 matrices applied to the input data during convolution.

6. ReLU (Rectified Linear Unit) activation function is used for the convolutional layers (`activation='relu'`). ReLU is common in CNNs because it introduces non-linearity and helps the model learn complex patterns in the data.

7. The `MaxPooling2D` layer reduces the spatial dimensions of the input data by taking the maximum value from each patch of the feature map. This helps in reducing computational complexity and controlling overfitting.

8. The `Flatten()` layer transforms the multi-dimensional output of the convolutional layers into a one-dimensional array, preparing it for input into the fully connected layers.

9. There are 64 neurons in the first `Dense` layer. This number may have been chosen based on the complexity of the task and the size of the dataset.

10. The last `Dense` layer uses the softmax activation function (`activation='softmax'`). This activation function is suitable for multi-class classification tasks as it converts the raw output into probabilities for each class.

11. The loss function specified is `'sparse_categorical_crossentropy'`, and the optimizer is `'adam'`. These choices are appropriate for multi-class classification tasks as they efficiently handle categorical data and optimize the model parameters.

12. During training, the model's performance is evaluated using accuracy as the metric. Accuracy measures the proportion of correctly classified samples.

13. The CNN is trained for 10 epochs, meaning the entire dataset is passed through the network 10 times during training.

14. The training datasets `X_train` and `y_train` are used for training the CNN.

15. The expected shape of the input data (`X_train`) for this CNN is `(batch_size, 32, 32, 3)`, where `batch_size` is the number of samples in each batch, and `32x32x3` represents the height, width, and number of channels (RGB) of the input images, respectively.
</details>