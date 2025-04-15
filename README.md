# 1. Multi Layer Perceptron
---

**AIM:**
To implement a Multilayer Perceptron (MLP) algorithm for the classification of handwritten digits from the MNIST dataset and evaluate its performance using metrics like accuracy.

---

**DESCRIPTION:**
The MNIST dataset is a widely-used benchmark for testing machine learning models. It consists of 70,000 grayscale images of size 28x28 pixels, representing handwritten digits (0-9). Multilayer Perceptrons are a class of artificial neural networks with fully connected layers. They learn to map input features to target outputs through layers of neurons, each utilizing activation functions to introduce non-linearities. In this experiment:
- The input images are preprocessed (normalized and flattened).
- The MLP architecture comprises an input layer, two hidden layers with ReLU activation, and an output layer with softmax activation.
- The model is trained using backpropagation and the Adam optimizer, with a categorical cross-entropy loss function.
- The performance is evaluated on test data.

---

**PROGRAM:**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # Convert labels to one-hot encoding
y_test = to_categorical(y_test, 10)

# Build the MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),   # Hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 neurons and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

---

_Example Output:_
```
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0821 - accuracy: 0.9753
313/313 [==============================] - 1s 2ms/step - loss: 0.0942 - accuracy: 0.9738
Test Accuracy: 0.9738
```

---
---

# 2. Multiclass classification
---

### **AIM:**  
To design a neural network for multiclass classification of newswires using the Reuters dataset and evaluate its performance.

---

### **DESCRIPTION:**  
The Reuters dataset contains thousands of newswire articles categorized into 46 different topics, making it a useful benchmark for multiclass classification. In this experiment, we'll preprocess the textual data by tokenizing and encoding it, followed by building a neural network using TensorFlow and Keras. The architecture will include an embedding layer to handle text input, followed by dense layers for classification.

---

### **PROGRAM:**

```python
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# Load Reuters dataset
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# Convert labels to one-hot encoding
num_classes = max(y_train) + 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(10000,)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

---

### **Output:**  
```
Epoch 10/10
313/313 [==============================] - 2s - loss: 0.4873 - accuracy: 0.8251
40/40 [==============================] - 1s - loss: 0.6402 - accuracy: 0.7967
Test Accuracy: 0.7967
```

---
---
# 3. One-Hot Encoding
---

### **AIM:**  
To implement One-Hot Encoding for words and characters in order to convert textual data into numerical representations suitable for machine learning models.

---

### **DESCRIPTION:**  
One-Hot Encoding is a technique used to transform categorical data into binary vectors, where each category (word or character) is represented as a unique binary array.  
- **Word-Level One-Hot Encoding:** Converts words into a sparse binary matrix, where each column represents a unique word in the vocabulary.  
- **Character-Level One-Hot Encoding:** Converts individual characters into unique binary vectors.  
- This method is commonly used in Natural Language Processing (NLP) tasks to prepare textual data for neural networks.

---

### **PROGRAM:**  

#### **1. One-Hot Encoding for Words**
```python
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample sentences
texts = ["hello world", "machine learning is great", "hello deep learning"]

# Initialize tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert words to one-hot encoded vectors
one_hot_results = tokenizer.texts_to_matrix(texts, mode="binary")

print("Word-Level One-Hot Encoding:")
print(one_hot_results)
```

---

#### **2. One-Hot Encoding for Characters**
```python
import numpy as np

# Sample text
text = "hello"

# Define vocabulary
vocab = sorted(set(text))  # Unique characters
char_to_index = {char: idx for idx, char in enumerate(vocab)}

# One-hot encoding for each character
one_hot_encoded = np.eye(len(vocab))[list(map(char_to_index.get, text))]

print("Character-Level One-Hot Encoding:")
print(one_hot_encoded)
```

---

### **OUTPUT:**  

#### **Example Output for Word-Level Encoding**
```
Word-Level One-Hot Encoding:
[[0. 1. 1. 0. 0. 0.]
 [0. 0. 0. 1. 1. 1.]
 [0. 1. 0. 1. 0. 1.]]
```
Each row represents a sentence, and each column a word in the vocabulary.

---

#### **Example Output for Character-Level Encoding**
For `"hello"`, assuming the vocabulary consists of `{'e', 'h', 'l', 'o'}`, the output might look like:
```
Character-Level One-Hot Encoding:
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```
Each row corresponds to a character encoded into its respective one-hot vector.

---
---
# 4. Convolutional Neural Network (CNN) 
---
### **AIM:**  
To build a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset and evaluate its performance.

---

### **DESCRIPTION:**  
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. Convolutional Neural Networks (CNNs) are a class of deep learning models designed specifically for image classification tasks. They use layers such as convolutional layers, pooling layers, and fully connected layers to automatically learn spatial hierarchies of features.  
- **Convolutional Layers** extract important patterns using filters.
- **Pooling Layers** reduce spatial dimensions to enhance computational efficiency.
- **Fully Connected Layers** classify the image into one of ten digit categories.  
In this experiment, we'll train a CNN model using TensorFlow/Keras and optimize it using the Adam optimizer with categorical cross-entropy loss.

---

### **PROGRAM:**  

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data to fit CNN input (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalize pixel values
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

---

### **OUTPUT:**  
After training for **10 epochs**, the CNN model achieves an accuracy of approximately **98-99%** on the MNIST dataset.

_Example Output:_  
```
Epoch 10/10
1875/1875 [==============================] - 15s - loss: 0.0284 - accuracy: 0.9914
313/313 [==============================] - 2s - loss: 0.0389 - accuracy: 0.9878
Test Accuracy: 0.9878
```

---
This CNN model efficiently extracts features and performs digit classification with high accuracy. You can experiment by modifying **filter sizes**, **number of layers**, and **epochs** to improve performance further.
---
---
# 5. VGG16
### **AIM:**  
To use a pre-trained Convolutional Neural Network (VGG16) for image classification and evaluate its performance on custom input images.

---

### **DESCRIPTION:**  
VGG16 is a well-known deep learning model trained on the ImageNet dataset, capable of classifying images into 1,000 categories. Instead of training a model from scratch, we leverage this pre-trained model to perform image classification efficiently.  
- The model consists of convolutional and fully connected layers that extract image features.
- The model weights are pre-trained on millions of images, ensuring high accuracy for common objects.
- We use TensorFlow/Keras to load the VGG16 model and classify images.

---

### **PROGRAM:**
```python
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load pre-trained VGG16 model (with ImageNet weights)
model = VGG16(weights='imagenet')

# Load and preprocess an image
image_path = 'your_image.jpg'  # Replace with actual image path
image = load_img(image_path, target_size=(224, 224))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)  # Normalize for VGG16

# Perform prediction
predictions = model.predict(image_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

# Display results
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}. {label} ({score:.2f})")
```

---

### **OUTPUT:**  
After running the program, the model predicts the top 3 categories for the given image.

_Example Output:_  
```
Predictions:
1. Labrador retriever (0.85)
2. Golden retriever (0.08)
3. Flat-coated retriever (0.03)
```
This output indicates that the model classifies the image as a **Labrador retriever** with an 85% confidence.

---
