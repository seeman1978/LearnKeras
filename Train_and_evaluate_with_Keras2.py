"""
Part I: Using built-in training & evaluation loops
When passing data to the built-in training loops of a model,
you should either use Numpy arrays (if your data is small and fits in memory)
or tf.data Dataset objects. In the next few paragraphs,
we'll use the MNIST dataset as Numpy arrays, in order to demonstrate how to use optimizers, losses, and metrics.
"""

import tensorflow as tf

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == "__main__":
    #we build in with the Functional API, but it could be a Sequential model or a subclassed model as well
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    #load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    #Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # List of metrics to monitor
                  metrics=['sparse_categorical_accuracy'])

    #Train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating over the entire dataset
    # for a given number of "epochs"
    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=3,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val))

    print('\nhistory dict:', history.history)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(x_test[:3])
    print('predictions shape:', predictions.shape)