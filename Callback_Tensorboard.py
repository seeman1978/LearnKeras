"""
Part I: Using built-in training & evaluation loops
When passing data to the built-in training loops of a model,
you should either use Numpy arrays (if your data is small and fits in memory)
or tf.data Dataset objects. In the next few paragraphs,
we'll use the MNIST dataset as Numpy arrays, in order to demonstrate how to use optimizers, losses, and metrics.
"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def get_compiled_model():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  return model

if __name__ == "__main__":
    model = get_compiled_model()

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

    # First, let's create a training Dataset instance.
    # For the sake of our example, we'll use the same MNIST data as before.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    # Now we get a test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)

    class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                    # Set weight "2" for class "5",
                    # making this class 2x more important
                    5: 2.,
                    6: 1., 7: 1., 8: 1., 9: 1.}

    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1),
        keras.callbacks.TensorBoard(log_dir='./full_path_to_your_logs',
              histogram_freq=0,  # How often to log histogram visualizations
              embeddings_freq=0,  # How often to log embedding visualizations
              update_freq='epoch')
    ]

    #Train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating over the entire dataset
    # for a given number of "epochs"
    # Since the dataset already takes care of batching,
    # we don't pass a `batch_size` argument.
    model.fit(train_dataset, class_weight=class_weight, epochs=3, validation_data=val_dataset, callbacks=callbacks)

    # You can also evaluate or predict on a dataset.
    print('\n# Evaluate')
    result = model.evaluate(test_dataset)
    dict(zip(model.metrics_names, result))