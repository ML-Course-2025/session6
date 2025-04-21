# Activity 2: Building and Comparing Image Classifiers (ANN vs. CNN)

---

## Part 1/3: MNIST Digit Classification

### Objective:

By the end of this activity, you will gain practical experience in building and training both Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) for image classification using the MNIST handwritten digit dataset. You will compare their performance and understand why CNNs are often preferred for image tasks.

Through this lab, you will achieve the following objectives:

1.  Load and preprocess the MNIST image dataset.
2.  Build, train, and evaluate a simple ANN model for image classification.
3.  Build, train, and evaluate a basic CNN model, applying concepts like Convolution, Pooling, and Flattening.
4.  Compare the performance of the ANN and CNN models using metrics and visualizations.
5.  Visualize model predictions to gain insights.

### Lab Steps:

**Step 1: Import Libraries**

*   We begin by importing the necessary tools: TensorFlow/Keras for building models, Matplotlib for plotting, NumPy for numerical operations, and Scikit-Learn for evaluation metrics.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
```

**Step 2: Load and Explore Dataset**

*   Load the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
*   Examine the shape of the data arrays (images and labels).
*   Display a few sample images with their corresponding labels.

```python
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

print("--- Dataset Shapes ---")
print("Training images shape:", X_train.shape) # Output: (60000, 28, 28) -> 60k images, 28x28 pixels each
print("Training labels shape:", y_train.shape) # Output: (60000,) -> 60k labels
print("Test images shape:", X_test.shape)     # Output: (10000, 28, 28)
print("Test labels shape:", y_test.shape)     # Output: (10000,)

print("\n--- Sample Images ---")
plt.figure(figsize=(10, 5))
for i in range(10): # Display first 10 images
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray') # Display in grayscale
    plt.title("Label: " + str(y_train[i]))
    plt.axis('off') # Hide axes
plt.tight_layout()
plt.show()
```

**Step 3: Preprocess Data**

*   **Normalization:** Neural networks generally perform better when input data is scaled. We normalize the pixel values from the range [0, 255] to [0, 1] by dividing by 255.0.
*   **Label Reshaping:** Ensure labels are 1D arrays (this is usually the default for `mnist.load_data` but good practice to confirm).

```python
print("\n--- Preprocessing ---")
# Check label shapes before potentially reshaping (optional, usually okay for MNIST)
print("Original y_train shape:", y_train.shape)
y_train = y_train.reshape(-1,) # Ensure y_train is a 1D array
y_test = y_test.reshape(-1,)   # Ensure y_test is a 1D array
print("Reshaped y_train shape:", y_train.shape)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0
print("Data normalized.")
```

**Step 4: Build and Train a Simple ANN Model**

*   We first build a basic ANN using `Flatten` to convert the 28x28 images into a 1D vector (784 elements) and `Dense` (fully connected) layers for classification.

```python
print("\n--- Building ANN Model ---")
ann = models.Sequential([
    # Flatten the 28x28 image into a 1D vector of 784 pixels
    layers.Flatten(input_shape=(28, 28)),

    # Dense hidden layer with 128 neurons and ReLU activation
    layers.Dense(128, activation='relu'),

    # Output layer with 10 neurons (one for each digit 0-9)
    # Softmax activation outputs probabilities for each class
    layers.Dense(10, activation='softmax')
])

# Compile the model: configure optimizer, loss function, and metrics
ann.compile(optimizer='adam',                     # Adam optimizer is generally a good default
            loss='sparse_categorical_crossentropy', # Use for integer labels (0, 1, ... 9)
            metrics=['accuracy'])                 # Track accuracy during training

ann.summary() # Print model architecture

print("\n--- Training ANN Model ---")
# Train the model for 5 epochs (passes through the entire training dataset)
history_ann = ann.fit(X_train, y_train, epochs=5, validation_split=0.1) # Use 10% of training data for validation
```

**Step 5: Evaluate the ANN Model**

*   Assess the trained ANN's performance on the unseen test data.
*   Generate a classification report (showing precision, recall, F1-score per class) and a confusion matrix (visualizing correct vs. incorrect predictions).

```python
print("\n--- Evaluating ANN Model ---")
loss_ann, accuracy_ann = ann.evaluate(X_test, y_test)
print(f"ANN Test Accuracy: {accuracy_ann:.4f}")

# Make predictions on the test set
y_pred_ann_prob = ann.predict(X_test)
# Convert probabilities to predicted class labels (index of the max probability)
y_pred_ann_classes = np.argmax(y_pred_ann_prob, axis=1)

# Print Classification Report
print("\nClassification Report (ANN):")
print(classification_report(y_test, y_pred_ann_classes))

# Display Confusion Matrix
print("\nConfusion Matrix (ANN):")
cm_ann = confusion_matrix(y_test, y_pred_ann_classes)
disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann)
disp_ann.plot(cmap=plt.cm.Blues)
plt.title("ANN Confusion Matrix")
plt.show()
```

**Step 6: Build and Train a CNN Model**

*   Now, build a CNN using `Conv2D` and `MaxPooling2D` layers to automatically extract spatial features before flattening and classifying with `Dense` layers.
*   **Important:** Keras `Conv2D` layers expect input data with a channel dimension. Since MNIST is grayscale, we reshape the (28, 28) images to (28, 28, 1).

```python
print("\n--- Building CNN Model ---")
cnn = models.Sequential([
    # First Convolutional Layer: Learns 32 basic features (like edges, corners)
    # kernel_size=(3,3): Uses a 3x3 filter window
    # activation='relu': Standard activation for non-linearity
    # input_shape=(28, 28, 1): MNIST image dimensions (Height, Width, Channels=1 for grayscale)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Pooling Layer: Reduces size by half (2x2 window), keeps strongest activations
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Layer: Learns 64 more complex features
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # Another pooling layer

    # Flatten Layer: Converts the 2D feature maps into a 1D vector
    layers.Flatten(),

    # Dense Layer: A fully connected layer for combining features
    layers.Dense(64, activation='relu'),

    # Output Layer: 10 neurons (one for each digit 0-9), softmax for probabilities
    layers.Dense(10, activation='softmax')
])

# Compile the CNN model
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.summary() # Print CNN architecture

# Reshape input data for CNN (add channel dimension)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
print("\nReshaped training data for CNN:", X_train_cnn.shape)

print("\n--- Training CNN Model ---")
# Train the CNN model (fewer epochs often needed than ANN for good performance)
history_cnn = cnn.fit(X_train_cnn, y_train, epochs=3, validation_split=0.1)
```

**Step 7: Evaluate the CNN Model**

*   Evaluate the trained CNN on the test data.

```python
print("\n--- Evaluating CNN Model ---")
loss_cnn, accuracy_cnn = cnn.evaluate(X_test_cnn, y_test)
print(f"CNN Test Accuracy: {accuracy_cnn:.4f}")
```

**Step 8: Compare Models and Visualize Predictions**

*   Compare the test accuracy of the ANN and CNN.
*   Visualize some predictions from the CNN model alongside the true labels.
*   Display the CNN's confusion matrix.

```python
print("\n--- Model Comparison ---")
print(f"ANN Test Accuracy: {accuracy_ann:.4f}")
print(f"CNN Test Accuracy: {accuracy_cnn:.4f}")
# Prompt for reflection:
print("\nNotice the difference in accuracy. CNNs typically excel at image tasks because")
print("convolutional layers effectively capture spatial hierarchies and features.")

# Make predictions with the CNN
y_pred_cnn_prob = cnn.predict(X_test_cnn)
y_pred_cnn_classes = np.argmax(y_pred_cnn_prob, axis=1)

# Define the enhanced plotting function
def plot_sample_with_pred(X, y_true, y_pred_classes, index, class_names=None):
    plt.figure(figsize=(3, 3))
    plt.imshow(X[index].reshape(28, 28), cmap='gray') # Reshape back for plotting if needed
    true_label = y_true[index]
    pred_label = y_pred_classes[index]
    title_str = f"True: {true_label}\nPred: {pred_label}"
    if class_names: # If class names list is provided
         title_str = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
    color = "green" if true_label == pred_label else "red"
    plt.title(title_str, color=color)
    plt.axis('off')
    plt.show()

print("\n--- Visualizing CNN Predictions (Examples) ---")
# Show prediction for a few test samples
plot_sample_with_pred(X_test, y_test, y_pred_cnn_classes, 0) # Index 0
plot_sample_with_pred(X_test, y_test, y_pred_cnn_classes, 5) # Index 5
plot_sample_with_pred(X_test, y_test, y_pred_cnn_classes, 10) # Index 10

# Display CNN Confusion Matrix
print("\nConfusion Matrix (CNN):")
cm_cnn = confusion_matrix(y_test, y_pred_cnn_classes)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)
disp_cnn.plot(cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.show()
```

**Step 9: Source Code Reference**

> The complete, runnable code for this part can be found here: [./src/lab2part1.py](./src/lab2part1.py).

---

## Part 2/3: CIFAR-10 Color Image Classification

### Objective:

This part extends the concepts to a more challenging dataset, CIFAR-10, which consists of small color images in 10 different classes. You will again build and compare ANN and CNN models.

Objectives:

1.  Load and preprocess the CIFAR-10 color image dataset.
2.  Define class names for better visualization.
3.  Build, train, and evaluate an ANN model.
4.  Build, train, and evaluate a CNN model suitable for color images.
5.  Compare performance and visualize results on this more complex dataset.

### Steps:

**Step 1: Import Libraries (if running separately)**

*   Ensure necessary libraries are imported (same as Part 1).

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
```

**Step 2: Load and Explore Dataset**

*   Load the CIFAR-10 dataset (50k training, 10k test images, 32x32 pixels, 3 color channels).
*   Define the class names corresponding to the integer labels (0-9).
*   Explore data shapes and display sample images with their class names.

```python
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = datasets.cifar10.load_data()

print("--- CIFAR-10 Dataset Shapes ---")
print("Training images shape:", X_train_cifar.shape) # Output: (50000, 32, 32, 3)
print("Training labels shape:", y_train_cifar.shape) # Output: (50000, 1) -> Note the extra dimension
print("Test images shape:", X_test_cifar.shape)     # Output: (10000, 32, 32, 3)
print("Test labels shape:", y_test_cifar.shape)     # Output: (10000, 1)

# Define class names for CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("\n--- CIFAR-10 Sample Images ---")
plt.figure(figsize=(10, 5))
for i in range(10): # Display first 10 images
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train_cifar[i])
    # Use y_train_cifar[i][0] because labels are shape (N, 1)
    plt.title(f"Class: {classes[y_train_cifar[i][0]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**Step 3: Preprocess Data**

*   **Reshape Labels:** Reshape the labels from (N, 1) to (N,) for compatibility with `sparse_categorical_crossentropy`.
*   **Normalization:** Normalize pixel values from [0, 255] to [0, 1].

```python
print("\n--- Preprocessing CIFAR-10 ---")
# Reshape labels to 1D arrays
y_train_cifar = y_train_cifar.reshape(-1,)
y_test_cifar = y_test_cifar.reshape(-1,)
print("Reshaped y_train_cifar shape:", y_train_cifar.shape)

# Normalize pixel values
X_train_cifar = X_train_cifar / 255.0
X_test_cifar = X_test_cifar / 255.0
print("CIFAR-10 data normalized.")
```

**Step 4: Build and Train a Simple ANN Model**

*   Build an ANN for CIFAR-10. Note that ANNs struggle significantly with the complexity of color images compared to simple digits. We use larger Dense layers here, but performance will likely be poor.

```python
print("\n--- Building ANN Model (CIFAR-10) ---")
ann_cifar = models.Sequential([
    # Flatten the 32x32x3 image into a 1D vector (3072 elements)
    layers.Flatten(input_shape=(32, 32, 3)),

    # Larger Dense hidden layers (ANNs need many parameters for images)
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),

    # Output layer with 10 neurons (one for each CIFAR class)
    layers.Dense(10, activation='softmax')
])

# Compile the model (using Adam for consistency)
ann_cifar.compile(optimizer='adam', # Changed from SGD to Adam
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

ann_cifar.summary()

print("\n--- Training ANN Model (CIFAR-10) ---")
# Train for a few epochs (ANN performance plateaus quickly here)
history_ann_cifar = ann_cifar.fit(X_train_cifar, y_train_cifar, epochs=5, validation_split=0.1)
```

**Step 5: Evaluate the ANN Model**

*   Evaluate the ANN on the CIFAR-10 test set. Expect relatively low accuracy.

```python
print("\n--- Evaluating ANN Model (CIFAR-10) ---")
loss_ann_cifar, accuracy_ann_cifar = ann_cifar.evaluate(X_test_cifar, y_test_cifar)
print(f"CIFAR-10 ANN Test Accuracy: {accuracy_ann_cifar:.4f}")

# Make predictions
y_pred_ann_cifar_prob = ann_cifar.predict(X_test_cifar)
y_pred_ann_cifar_classes = np.argmax(y_pred_ann_cifar_prob, axis=1)

# Print Classification Report
print("\nClassification Report (CIFAR-10 ANN):")
# Use target_names for readable labels
print(classification_report(y_test_cifar, y_pred_ann_cifar_classes, target_names=classes))

# Display Confusion Matrix
print("\nConfusion Matrix (CIFAR-10 ANN):")
cm_ann_cifar = confusion_matrix(y_test_cifar, y_pred_ann_cifar_classes)
disp_ann_cifar = ConfusionMatrixDisplay(confusion_matrix=cm_ann_cifar, display_labels=classes)
disp_ann_cifar.plot(cmap=plt.cm.Blues, xticks_rotation='vertical') # Rotate labels if needed
plt.title("CIFAR-10 ANN Confusion Matrix")
plt.tight_layout()
plt.show()
```

**Step 6: Build and Train a CNN Model**

*   Build a CNN suitable for the 32x32x3 CIFAR-10 images. This architecture is similar to the MNIST CNN but adjusted for the input shape and potentially needing more training.

```python
print("\n--- Building CNN Model (CIFAR-10) ---")
cnn_cifar = models.Sequential([
    # Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation. Input shape is 32x32x3
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    # Pool Layer 1: Max pooling with 2x2 window
    layers.MaxPooling2D((2, 2)),

    # Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # Pool Layer 2: Max pooling with 2x2 window
    layers.MaxPooling2D((2, 2)),

    # Flatten the feature maps
    layers.Flatten(),

    # Dense Layer: 64 neurons, ReLU activation
    layers.Dense(64, activation='relu'),
    # Output Layer: 10 neurons (for 10 classes), softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the CNN model
cnn_cifar.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_cifar.summary()

print("\n--- Training CNN Model (CIFAR-10) ---")
# Train for more epochs as CIFAR-10 is more complex than MNIST
history_cnn_cifar = cnn_cifar.fit(X_train_cifar, y_train_cifar, epochs=10, validation_split=0.1)
```

**Step 7: Evaluate the CNN Model**

*   Evaluate the trained CNN on the CIFAR-10 test set.

```python
print("\n--- Evaluating CNN Model (CIFAR-10) ---")
loss_cnn_cifar, accuracy_cnn_cifar = cnn_cifar.evaluate(X_test_cifar, y_test_cifar)
print(f"CIFAR-10 CNN Test Accuracy: {accuracy_cnn_cifar:.4f}")
```

**Step 8: Compare Models and Visualize Predictions**

*   Compare ANN and CNN accuracy on CIFAR-10.
*   Visualize some CNN predictions.
*   Display the CNN's confusion matrix.

```python
print("\n--- CIFAR-10 Model Comparison ---")
print(f"CIFAR-10 ANN Test Accuracy: {accuracy_ann_cifar:.4f}")
print(f"CIFAR-10 CNN Test Accuracy: {accuracy_cnn_cifar:.4f}")
# Prompt for reflection:
print("\nThe performance gap between ANN and CNN is usually even larger on")
print("more complex datasets like CIFAR-10 compared to MNIST.")

# Make predictions with the CNN
y_pred_cnn_cifar_prob = cnn_cifar.predict(X_test_cifar)
y_pred_cnn_cifar_classes = np.argmax(y_pred_cnn_cifar_prob, axis=1)

# Define the enhanced plotting function (can reuse from Part 1 if in same notebook)
def plot_sample_with_pred_cifar(X, y_true, y_pred_classes, index, class_names):
    plt.figure(figsize=(3, 3))
    # Display color image - ensure pixel values are appropriate (e.g., 0-1 or 0-255)
    # If normalized to 0-1, imshow works fine. If 0-255, use imshow(X[index].astype('uint8'))
    plt.imshow(X[index])
    true_label = y_true[index]
    pred_label = y_pred_classes[index]
    title_str = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
    color = "green" if true_label == pred_label else "red"
    plt.title(title_str, color=color)
    plt.axis('off')
    plt.show()


print("\n--- Visualizing CNN Predictions (CIFAR-10 Examples) ---")
# Show prediction for a few test samples
plot_sample_with_pred_cifar(X_test_cifar, y_test_cifar, y_pred_cnn_cifar_classes, 3, classes) # Index 3
plot_sample_with_pred_cifar(X_test_cifar, y_test_cifar, y_pred_cnn_cifar_classes, 7, classes) # Index 7
plot_sample_with_pred_cifar(X_test_cifar, y_test_cifar, y_pred_cnn_cifar_classes, 9, classes) # Index 9


# Display CNN Confusion Matrix
print("\nConfusion Matrix (CIFAR-10 CNN):")
cm_cnn_cifar = confusion_matrix(y_test_cifar, y_pred_cnn_cifar_classes)
disp_cnn_cifar = ConfusionMatrixDisplay(confusion_matrix=cm_cnn_cifar, display_labels=classes)
disp_cnn_cifar.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("CIFAR-10 CNN Confusion Matrix")
plt.tight_layout()
plt.show()
```

**Step 9: Source Code Reference**

> The complete, runnable code for this part can be found here: [./src/lab2part2.py](./src/lab2part2.py).

---

## Part 3/3: Discussion

*   Refer to this specific code snippet, which defines the CNN model used in **Part 2 (CIFAR-10), Step 6**:

```python
# CNN Model for CIFAR-10
cnn_cifar = models.Sequential([
    # Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation. Input shape is 32x32x3
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    # Pool Layer 1: Max pooling with 2x2 window
    layers.MaxPooling2D((2, 2)),

    # Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # Pool Layer 2: Max pooling with 2x2 window
    layers.MaxPooling2D((2, 2)),

    # Flatten the feature maps
    layers.Flatten(),

    # Dense Layer: 64 neurons, ReLU activation
    layers.Dense(64, activation='relu'),
    # Output Layer: 10 neurons (for 10 classes), softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the CNN model
cnn_cifar.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Training command (context for Q13-15)
# history_cnn_cifar = cnn_cifar.fit(X_train_cifar, y_train_cifar, epochs=10)
```

**Discussion Questions:**

1.  What type of neural network architecture is being defined in this code snippet?
2.  What role does the `models.Sequential([...])` structure play in this code?
3.  How many convolutional layers (`Conv2D`) are included in this network?
4.  What is the main purpose of the `Conv2D` layers in a CNN, based on the summary?
5.  What does the `kernel_size=(3, 3)` parameter specify for the `Conv2D` layers?
6.  What activation function is used *after* the convolutions (`Conv2D`) and in the first `Dense` layer? Why is `relu` a common choice here?
7.  How does the `MaxPooling2D((2, 2))` layer affect the feature maps passing through it? What are the benefits mentioned in the summary?
8.  What is the purpose of the `Flatten()` layer in this architecture? Why is it necessary?
9.  How many neurons are in the first `Dense` layer (after flattening)? Is this number arbitrary, or how might it be chosen?
10. What activation function is used in the *final* `Dense` layer? Why is `softmax` suitable for this multi-class classification task?
11. The model is compiled with `loss='sparse_categorical_crossentropy'` and `optimizer='adam'`. Why are these appropriate choices for this specific task (CIFAR-10 classification with integer labels)?
12. The `metrics=['accuracy']` parameter is included during compilation. What does 'accuracy' measure, and why is it a relevant metric here?
13. If the model were trained using `epochs=10` (as shown in the commented training line), what does this mean?
14. Based on the context of Part 2, what specific datasets would typically be passed as `X_train_cifar` and `y_train_cifar` to the `fit` method?
15. What is the exact, expected shape of a single input sample (e.g., one image from `X_train_cifar`) that this CNN model takes, according to the `input_shape` parameter in the first `Conv2D` layer?

<details>
<summary>Sample answers</summary>

1.  The code defines a **Convolutional Neural Network (CNN)** architecture.
2.  `models.Sequential([...])` creates a linear stack of layers, where the output of one layer feeds directly into the next layer in the defined order.
3.  There are **two** convolutional layers (`Conv2D`) in this CNN.
4.  The main purpose of `Conv2D` layers is to **apply convolutional filters** to the input (image or feature maps) to automatically **detect and extract spatial features** (like edges, textures, patterns).
5.  `kernel_size=(3, 3)` specifies the dimensions (height and width) of the **convolutional filter (or kernel)** used to scan the input. It's a 3x3 window.
6.  The **ReLU (Rectified Linear Unit)** activation function (`activation='relu'`) is used. It's common because it introduces **non-linearity** efficiently (allowing the model to learn complex relationships) and helps mitigate the vanishing gradient problem compared to older functions like sigmoid or tanh in deep networks.
7.  `MaxPooling2D((2, 2))` **reduces the spatial dimensions** (height and width) of the feature maps, typically by half if the stride equals the pool size (which is the default). It does this by taking the maximum value within each 2x2 window. Benefits include **reducing computational cost/parameters** and providing some **translation invariance** (making the model less sensitive to the exact location of features).
8.  The `Flatten()` layer **transforms the multi-dimensional output** (e.g., height x width x channels) of the preceding convolutional/pooling layers into a **one-dimensional vector**. This is necessary because the subsequent `Dense` (fully connected) layers require 1D vector input.
9.  There are **64 neurons** in the first `Dense` layer. The number is a hyperparameter; it's often chosen based on experimentation, balancing model capacity (ability to learn complex patterns) against the risk of overfitting and computational cost. It doesn't have a fixed theoretical value but is typically reduced compared to the flattened vector size.
10. The **Softmax** activation function (`activation='softmax'`) is used in the final `Dense` layer. It's suitable because it converts the raw outputs (logits) of the 10 neurons into a **probability distribution**, where each output represents the probability of the input belonging to one of the 10 classes, and all probabilities sum to 1.
11. `loss='sparse_categorical_crossentropy'` is appropriate because it's designed for **multi-class classification** where the true labels (`y_train_cifar`) are provided as **integers** (0, 1, ..., 9). `optimizer='adam'` is a widely used, effective, and generally robust optimization algorithm that adapts the learning rate during training, often leading to good convergence.
12. `accuracy` measures the **proportion (or percentage) of input samples that are correctly classified** by the model during training and evaluation. It's relevant because it provides a straightforward, interpretable measure of the model's overall classification performance.
13. `epochs=10` means the entire training dataset (`X_train_cifar`, `y_train_cifar`) is **passed forward and backward through the neural network 10 times** during the training process.
14. `X_train_cifar` would contain the normalized CIFAR-10 training images (shape `(50000, 32, 32, 3)`), and `y_train_cifar` would contain the corresponding integer labels (0-9) for those images (shape `(50000,)`).
15. The expected shape of a single input sample is `(32, 32, 3)`, representing **Height=32 pixels, Width=32 pixels, and Channels=3** (for RGB color), as specified by `input_shape=(32, 32, 3)` in the first `Conv2D` layer.

</details>

