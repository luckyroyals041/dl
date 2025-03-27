# Basic Operations on Tensorflow

## Aim: A Program of Basic Operations on TensorFlow

## Description:
TensorFlow is an open-source machine learning library developed by Google. TensorFlow is used to build and train deep learning models as it facilitates the creation of computational graphs and efficient execution on various hardware platforms. The article provides a comprehensive overview of TensorFlow.

TensorFlow is basically a software library for numerical computation using data flow graphs where:
- **Nodes** in the graph represent mathematical operations.
- **Edges** in the graph represent the multidimensional data arrays (called tensors) communicated between them. *(Please note that tensor is the central unit of data in TensorFlow)*

### Tensor Operations:
- **Tensor Addition**: You can add two tensors using `tensorA.add(tensorB)`
- **Tensor Subtraction**: You can subtract two tensors using `tensorA.sub(tensorB)`
- **Tensor Multiplication**: You can multiply two tensors using `tensorA.mul(tensorB)`
- **Tensor Division**: You can divide two tensors using `tensorA.div(tensorB)`
- **Tensor Square**: You can square a tensor using `tensor.square()`
- **Tensor Reshape**: The number of elements in a tensor is the product of the sizes in the shape. Since there can be different shapes with the same size, it is often useful to reshape a tensor to other shapes with the same size. You can reshape a tensor using `tensor.reshape()`

```python
#Basic Operations on Tensor:
import tensorflow as tf
# Create two 2D tensors
t1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
t2 = tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.float32)
# Perform basic operations
#Addition
add = tf.add(t1,t2)
print("\nAddition Result:\n", add)
#Substraction
sub = tf.subtract(t1, t2)
print("\nSubtraction Result:\n", sub)
#Multiplication
mul = tf.multiply(t1,t2)
print("\nMultiplication Result:\n", mul)
#Division
div = tf.divide(t1, t2)
print("\nDivision Result:\n", div)
#Power
pow = tf.pow(t1, 2)
print("\nElement-wise Power (Tensor A^2):\n", pow)

#Transpose
transpose_t1 = tf.transpose(t1)
print("\nTranspose of Tensor A:\n", transpose_t1)
#Matrix Multiplication
matmul = tf.matmul(t1, tf.transpose(t1))
print("\nMatrix Multiplication Result (A @ B^T):\n", matmul)
```

Output:
```

Addition Result:
 tf.Tensor(
[[ 8. 10. 12.]
 [14. 16. 18.]], shape=(2, 3), dtype=float32)

Subtraction Result:
 tf.Tensor(
[[-6. -6. -6.]
 [-6. -6. -6.]], shape=(2, 3), dtype=float32)

Multiplication Result:
 tf.Tensor(
[[ 7. 16. 27.]
 [40. 55. 72.]], shape=(2, 3), dtype=float32)

Division Result:
 tf.Tensor(
[[0.14285715 0.25       0.33333334]
 [0.4        0.45454547 0.5       ]], shape=(2, 3), dtype=float32)

Element-wise Power (Tensor A^2):
 tf.Tensor(
[[ 1.  4.  9.]
 [16. 25. 36.]], shape=(2, 3), dtype=float32)

Transpose of Tensor A:
 tf.Tensor(
[[1. 4.]
 [2. 5.]
 [3. 6.]], shape=(3, 2), dtype=float32)

Matrix Multiplication Result (A @ B^T):
 tf.Tensor(
[[14. 32.]
 [32. 77.]], shape=(2, 2), dtype=float32)

```
# Binary Classification
## Aim: Design a Neural Network for Classifying Movie Reviews (Binary Classification)

## Description:
### Neural Network Design:
- **Dataset**: IMDB Movie Reviews Dataset (50,000 reviews, balanced across positive and negative sentiments).
- **Preprocessing**: 
  - Tokenization
  - Padding
  - Sequence truncation
- **Model Architecture**:
  1. **Embedding Layer**: Converts words into dense vectors.
  2. **LSTM Layer**: Captures sequential dependencies in the text.
  3. **Dense Layers**: Fully connected layers for classification.
  4. **Output Layer**: Sigmoid activation for binary classification.

```python
# Import required libraries
#Binary classification
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, losses, metrics

# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

# Define vectorization function
def vectorize_sequences(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Vectorize training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convert labels to float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Create the model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)

# Prepare validation data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# Get training history
history_dict = history.history

# Plot training and validation loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.figure(1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

# Retrain the model with fewer epochs
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=3,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# Plot training and validation accuracy
plt.figure(2)
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions
np.set_printoptions(suppress=True)
result = model.predict(x_test)
print(result)

```

Output:
```
/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 35ms/step - binary_accuracy: 0.6396 - loss: 0.6308 - val_binary_accuracy: 0.7857 - val_loss: 0.4741
Epoch 2/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - binary_accuracy: 0.8188 - loss: 0.4299 - val_binary_accuracy: 0.8357 - val_loss: 0.3849
Epoch 3/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 36ms/step - binary_accuracy: 0.8489 - loss: 0.3622 - val_binary_accuracy: 0.8485 - val_loss: 0.3543
Epoch 4/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - binary_accuracy: 0.8685 - loss: 0.3245 - val_binary_accuracy: 0.8484 - val_loss: 0.3549
Epoch 5/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - binary_accuracy: 0.8737 - loss: 0.3117 - val_binary_accuracy: 0.8490 - val_loss: 0.3549
Epoch 6/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 32ms/step - binary_accuracy: 0.8770 - loss: 0.2983 - val_binary_accuracy: 0.8598 - val_loss: 0.3369
Epoch 7/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - binary_accuracy: 0.8840 - loss: 0.2934 - val_binary_accuracy: 0.8570 - val_loss: 0.3395
Epoch 8/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step - binary_accuracy: 0.8830 - loss: 0.2807 - val_binary_accuracy: 0.8584 - val_loss: 0.3448
Epoch 9/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - binary_accuracy: 0.8842 - loss: 0.2831 - val_binary_accuracy: 0.8592 - val_loss: 0.3393
Epoch 10/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - binary_accuracy: 0.8881 - loss: 0.2757 - val_binary_accuracy: 0.8585 - val_loss: 0.3395
Epoch 11/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - binary_accuracy: 0.8896 - loss: 0.2688 - val_binary_accuracy: 0.8556 - val_loss: 0.3456
Epoch 12/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - binary_accuracy: 0.8882 - loss: 0.2689 - val_binary_accuracy: 0.8493 - val_loss: 0.3595
Epoch 13/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - binary_accuracy: 0.8954 - loss: 0.2576 - val_binary_accuracy: 0.8578 - val_loss: 0.3452
Epoch 14/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - binary_accuracy: 0.8919 - loss: 0.2565 - val_binary_accuracy: 0.8547 - val_loss: 0.3512
Epoch 15/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - binary_accuracy: 0.8925 - loss: 0.2624 - val_binary_accuracy: 0.8557 - val_loss: 0.3468
Epoch 16/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - binary_accuracy: 0.8958 - loss: 0.2538 - val_binary_accuracy: 0.8513 - val_loss: 0.3639
Epoch 17/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - binary_accuracy: 0.8995 - loss: 0.2489 - val_binary_accuracy: 0.8456 - val_loss: 0.3695
Epoch 18/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - binary_accuracy: 0.9036 - loss: 0.2391 - val_binary_accuracy: 0.8521 - val_loss: 0.3581
Epoch 19/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - binary_accuracy: 0.9056 - loss: 0.2332 - val_binary_accuracy: 0.8518 - val_loss: 0.3568
Epoch 20/20
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - binary_accuracy: 0.9025 - loss: 0.2330 - val_binary_accuracy: 0.8477 - val_loss: 0.3756
<Figure size 640x480 with 1 Axes>Epoch 1/3
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - binary_accuracy: 0.9103 - loss: 0.2320 - val_binary_accuracy: 0.8448 - val_loss: 0.3911
Epoch 2/3
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - binary_accuracy: 0.9065 - loss: 0.2297 - val_binary_accuracy: 0.8518 - val_loss: 0.3616
Epoch 3/3
[1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - binary_accuracy: 0.9126 - loss: 0.2152 - val_binary_accuracy: 0.8511 - val_loss: 0.3637
<Figure size 640x480 with 1 Axes>[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step
[[0.2522238]
 [0.9939799]
 [0.7916688]
 ...
 [0.0641171]
 [0.0416252]
 [0.1762261]]

```
# Boston Housing Price
## Aim: Design a Neural Network for Predicting House Prices Using Boston Housing Price Dataset

## Description:
A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain.

### Neural Network Structure:
A neural network is composed of:
1. **Input Layer**: Receives the raw input data.
2. **Hidden Layers**: One or more intermediate layers between the input and output layers. These layers perform feature extraction and transformation.
3. **Output Layer**: Produces the final output of the network.

The two stages of the basic process are:
- **Forward Propagation**: Data flows forward through the network to generate predictions.
- **Backpropagation**: Errors are propagated backward to adjust weights and improve accuracy.

### Key Features:
- **Layers**:
  - Input Layer: Accepts the input data.
  - Hidden Layers: Capture and transform features.
  - Output Layer: Provides the output of the network.

### Dataset:
The Boston Housing Price dataset (also referred to as Boston Housing Dataset) is commonly used in machine learning and statistics. This dataset was first introduced by Harrison and Rubinfeld in 1978 and is often utilized for regression analysis tasks. It contains information useful for predicting house prices in the Boston area.



```python
#Boston Housing Price.
import numpy as np
import pandas as pd
import tensorflow as tf
H = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=H.load_data()
train_x.shape
#build neural network
model=tf.keras.Sequential([
tf.keras.layers.Dense(13,activation='relu'),
tf.keras.layers.Dense(6,activation='relu'),
tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(train_x,train_y,epochs=13)
#evaluate
model.evaluate(test_x,test_y)
```

Output:
```
Epoch 1/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - loss: 602.2186 - mae: 22.5729
Epoch 2/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 605.6759 - mae: 22.8874 
Epoch 3/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 584.5156 - mae: 22.4413 
Epoch 4/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 588.7748 - mae: 22.4163 
Epoch 5/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 601.9792 - mae: 22.5748 
Epoch 6/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 542.6711 - mae: 21.5644 
Epoch 7/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 596.6973 - mae: 22.5453 
Epoch 8/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 583.6337 - mae: 22.3483 
Epoch 9/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 582.8398 - mae: 22.3972 
Epoch 10/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 620.7535 - mae: 23.1191 
Epoch 11/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 562.3947 - mae: 21.8854 
Epoch 12/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 565.7959 - mae: 21.9868 
Epoch 13/13
[1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 609.5273 - mae: 22.7640 
[1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 586.1717 - mae: 22.5703  
[615.8577880859375, 23.078432083129883]
```

# Word Embeddings
## Aim: Implement Word Embeddings for IMDB Dataset

## Description:
Word embeddings for the IMDB dataset involve representing words as dense vectors in a continuous vector space, capturing semantic relationships between words. The IMDB dataset is commonly used for sentiment analysis, where the goal is to classify movie reviews as either positive or negative.

### Popular Methods for Generating Word Embeddings:
1. **Word2Vec**: Learns vector representations by predicting the surrounding words in a large text corpus.
2. **GloVe (Global Vectors for Word Representation)**: Constructs word embeddings based on co-occurrence statistics in the corpus.

### Benefits of Word Embeddings:
By using word embeddings:
- The model can capture semantic similarities between words.
- It generalizes better to unseen data.
- Improved sentiment analysis performance is achieved on the IMDB dataset.



```python
#word embedding
from keras.datasets import imdb
from keras import preprocessing
max_features=10000
max_len=20
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train.shape
x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=max_len)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=max_len)
x_train.shape
from keras.models import Sequential
from keras.layers import Flatten,Dense
from tensorflow.keras.layers import Embedding
model=Sequential()
model.add(Embedding(10000,8,input_length=max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train, epochs=10, batch_size=32, validation_split=0.2)
```

Output:
```
[1mModel: "sequential_7"[0m
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mLayer (type)                        [0m[1m [0m┃[1m [0m[1mOutput Shape               [0m[1m [0m┃[1m [0m[1m        Param #[0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding_2 ([38;5;33mEmbedding[0m)              │ ?                           │     [38;5;34m0[0m (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_2 ([38;5;33mFlatten[0m)                  │ ?                           │     [38;5;34m0[0m (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_17 ([38;5;33mDense[0m)                     │ ?                           │     [38;5;34m0[0m (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
[1m Total params: [0m[38;5;34m0[0m (0.00 B)
[1m Trainable params: [0m[38;5;34m0[0m (0.00 B)
[1m Non-trainable params: [0m[38;5;34m0[0m (0.00 B)
Epoch 1/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 5ms/step - accuracy: 0.5627 - loss: 0.6858 - val_accuracy: 0.6952 - val_loss: 0.6193
Epoch 2/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 3ms/step - accuracy: 0.7410 - loss: 0.5700 - val_accuracy: 0.7302 - val_loss: 0.5308
Epoch 3/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 5ms/step - accuracy: 0.7841 - loss: 0.4710 - val_accuracy: 0.7458 - val_loss: 0.5034
Epoch 4/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8044 - loss: 0.4305 - val_accuracy: 0.7512 - val_loss: 0.4965
Epoch 5/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.8253 - loss: 0.3961 - val_accuracy: 0.7540 - val_loss: 0.4944
Epoch 6/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 3ms/step - accuracy: 0.8411 - loss: 0.3712 - val_accuracy: 0.7518 - val_loss: 0.4968
Epoch 7/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 5ms/step - accuracy: 0.8496 - loss: 0.3529 - val_accuracy: 0.7544 - val_loss: 0.5033
Epoch 8/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.8589 - loss: 0.3324 - val_accuracy: 0.7558 - val_loss: 0.5106
Epoch 9/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 3ms/step - accuracy: 0.8694 - loss: 0.3140 - val_accuracy: 0.7514 - val_loss: 0.5185
Epoch 10/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.8784 - loss: 0.2972 - val_accuracy: 0.7476 - val_loss: 0.5270

```

# RNN
## Aim: Implement a Recurrent Neural Network for IMDB Movie Review Classification Problem

## Description:
A Recurrent Neural Network (RNN) is a type of neural network where the output of the previous step is fed back as input to the current step. This structure allows RNNs to maintain a memory of past information in their hidden state, which helps the model understand sequential data.

### Key Features of RNNs:
1. **Hidden State**:
   - The hidden state is a crucial part of RNNs as it retains information from previous inputs.
   - In traditional neural networks, each input is independent. However, for tasks like language modeling, where context from previous words is important, RNNs help by passing the hidden state from one time step to the next.

2. **Memory State**:
   - The memory state (hidden state) stores information about previous inputs in the sequence.
   - This allows the network to "remember" past data, such as words in a sentence, for making predictions on future steps.

3. **Parameter Sharing**:
   - Unlike traditional neural networks that use different parameters for each input, RNNs share the same parameters at every time step.
   - This reduces complexity and makes RNNs more efficient for sequential tasks.

---

```python
# Word Embedding with IMDB Dataset
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.layers import Embedding
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

# Define constants
max_features = 10000
maxlen = 500
batch_size = 32

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# Pad sequences
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Build the model
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=batch_size,
    validation_split=0.2
)

# Extract training history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Plot training and validation accuracy
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

```

Output:
```
Training data shape: (25000, 20)
[1mModel: "sequential_9"[0m
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃[1m [0m[1mLayer (type)                        [0m[1m [0m┃[1m [0m[1mOutput Shape               [0m[1m [0m┃[1m [0m[1m        Param #[0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding_4 ([38;5;33mEmbedding[0m)              │ ?                           │     [38;5;34m0[0m (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_4 ([38;5;33mFlatten[0m)                  │ ?                           │     [38;5;34m0[0m (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_19 ([38;5;33mDense[0m)                     │ ?                           │     [38;5;34m0[0m (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
[1m Total params: [0m[38;5;34m0[0m (0.00 B)
[1m Trainable params: [0m[38;5;34m0[0m (0.00 B)
[1m Non-trainable params: [0m[38;5;34m0[0m (0.00 B)
Epoch 1/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 4ms/step - accuracy: 0.5765 - loss: 0.6838 - val_accuracy: 0.7026 - val_loss: 0.6070
Epoch 2/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.7441 - loss: 0.5603 - val_accuracy: 0.7316 - val_loss: 0.5218
Epoch 3/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 4ms/step - accuracy: 0.7850 - loss: 0.4719 - val_accuracy: 0.7444 - val_loss: 0.5003
Epoch 4/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8082 - loss: 0.4267 - val_accuracy: 0.7494 - val_loss: 0.4969
Epoch 5/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.8187 - loss: 0.4009 - val_accuracy: 0.7526 - val_loss: 0.4966
Epoch 6/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 5ms/step - accuracy: 0.8337 - loss: 0.3796 - val_accuracy: 0.7530 - val_loss: 0.5001
Epoch 7/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8427 - loss: 0.3623 - val_accuracy: 0.7498 - val_loss: 0.5052
Epoch 8/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8542 - loss: 0.3404 - val_accuracy: 0.7486 - val_loss: 0.5094
Epoch 9/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.8611 - loss: 0.3266 - val_accuracy: 0.7486 - val_loss: 0.5158
Epoch 10/10
[1m625/625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 5ms/step - accuracy: 0.8757 - loss: 0.3005 - val_accuracy: 0.7454 - val_loss: 0.5225

```

```python
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

# Define constants
max_features = 10000
maxlen = 500
batch_size = 32

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# Pad sequences
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Build the model
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=batch_size,
    validation_split=0.2
)

# Extract training history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Plot training and validation accuracy
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

```

Output:
```
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 500)
x_test shape: (25000, 500)
Epoch 1/10
[1m480/625[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m19s[0m 132ms/step - accuracy: 0.5841 - loss: 0.6542
```

