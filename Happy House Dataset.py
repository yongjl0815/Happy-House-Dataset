import pandas as pd
import numpy as np

import os
print(os.listdir("../input"))

# To load h5 file 
import h5py

# Tensorflow and Keras
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPool2D, Flatten, Dense

# For checking accuracy, etc.
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Load Data
def load_dataset(path_to_train, path_to_test):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y

# Get train and test data
X_train, Y_train, X_test, Y_test = load_dataset("../input/train_happy.h5", "../input/test_happy.h5")

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# Rescaling data
X_train = X_train/255.
X_test = X_test/255.
Y_train = Y_train.T
Y_test = Y_test.T

# Initialize CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation ='relu'))

# Pooling
model.add(MaxPool2D(pool_size=(2,2)))

# Flattening
model.add(Flatten())

# Connect convolutional network to neural network
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compile
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit
model.fit(X_train, Y_train, batch_size=30, epochs=20)


# Test
Y_predict = model.predict_classes(X_test)

#Y_predict


# Check how good the model is
accuracy = accuracy_score(Y_test, Y_predict)
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, Y_predict, average='binary')
 
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
