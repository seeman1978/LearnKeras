"""
The functional API is a way to build graphs of layers.
"""
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

if __name__ == "__main__":
    #a deep learning model is usually a directed acyclic graph (DAG) of layers. 有向无环图
    inputs = keras.Input(shape=(784,))
    dense = layers.Dense(64, activation='relu')
    x = dense(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    # create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    model.summary()

    #需要在操作系统中安装graphviz。 sudo apt-get install graphviz
    keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=5,
                        validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    model.save('my_model')

    # Recreate the exact same model purely from the file:
    model2 = keras.models.load_model('my_model')

    #Use the same graph of layers to define multiple models
    encoder_input = keras.Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    encoder.summary()

    x = layers.Reshape((4, 4, 1))(encoder_output)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    autoencoder.summary()
    #The reverse of a Conv2D layer is a Conv2DTranspose layer, and the reverse of a MaxPooling2D layer is an UpSampling2D layer.