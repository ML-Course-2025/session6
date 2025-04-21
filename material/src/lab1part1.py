import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

# Load datasets
cereal_data = pd.read_csv("cereal.csv")

# Preprocess datasets
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

# Convert non-numeric values to NaN and then fill or drop them
cereal_features = cereal_features.apply(pd.to_numeric, errors='coerce')
cereal_features.dropna(inplace=True)
cereal_target = cereal_target[cereal_features.index]

# Split datasets into training and testing sets
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(cereal_features, cereal_target, test_size=0.2, random_state=42)

# Define a Sequential model
model = Sequential()

# Add dense layers
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print model summary
model.summary()

# Train the model
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=10, batch_size=64, validation_data=(cereal_X_test, cereal_y_test), verbose=1)

# Plot training history
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
