import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
        model.add(tf.keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.SGD(0.2)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=optimizer)

    model.summary()

    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(32)

    model.fit(dataset, epochs=1)