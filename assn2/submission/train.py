import numpy as np
import os
from tensorflow.keras import layers, Sequential, regularizers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file



# Load the data into numpy arrays
def loadData( filename, dictSize = 225 ):
	X, y = load_svmlight_file( filename, multilabel = False, n_features = dictSize, offset = 1 )
	return (X, y)

X,y = loadData("train")
X = X.toarray()
y = y - 1               # convert class labels to 0 to 49


# validation split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# Build the neural network model
model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(225,)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(50, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs = 300, batch_size=512, validation_data=(X_test, y_test))

# save the model
model.save('model.h5')
