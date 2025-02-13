# MNIST Handwritten Digit Classification

## Overview
This project implements a neural network to classify handwritten digits from the MNIST dataset using Keras and TensorFlow. The model is trained using a fully connected neural network and evaluated for accuracy.

## Installation
To run this project, install the necessary dependencies:
```bash
%pip install scikit-learn keras matplotlib seaborn
```

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels.

## Steps
### 1. Import Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
```

### 2. Load Dataset
```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### 3. Data Visualization
A few sample images from the dataset are displayed using Matplotlib.

### 4. Data Preprocessing
- Normalize pixel values to the range [0,1]
- Flatten images into a 784-dimensional vector
- Convert labels to one-hot encoding

### 5. Define and Compile Model
A sequential neural network with two hidden layers (128 neurons each) and a softmax output layer is defined and compiled.
```python
model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 6. Training
The model is trained using 512 batch size for 20 epochs.
```python
batch_size = 512
epochs = 20
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
```

### 7. Evaluation
The model is evaluated on test data to compute accuracy and loss.
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
```

### 8. Predictions
Predictions are generated and displayed for a random test sample.
```python
random_idx = np.random.choice(len(x_test))
y_sample_pred_class = np.argmax(model.predict(x_test[random_idx:random_idx+1]))
plt.imshow(x_test[random_idx].reshape(28,28), cmap='gray')
plt.title(f'Predicted: {y_sample_pred_class}, True: {np.argmax(y_test[random_idx])}')
```

## Results
- The model achieves high accuracy (~97-99%) on the test dataset.
- Predictions for random samples are visualized.
- The confusion matrix can be generated for further analysis.


