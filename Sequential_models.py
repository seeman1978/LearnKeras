"""
TensorFlow 2 quickstart for beginners
This short introduction uses Keras to:

Build a neural network that classifies images.
Train this neural network.
And, finally, evaluate the accuracy of the model.

"""
import tensorflow as tf

if __name__== "__main__":
    """
    Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    # Save entire model to a HDF5 file
    model.save('my_model')  #The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration.

    # Recreate the exact same model, including weights and optimizer.
    model = tf.keras.models.load_model('my_model')

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    probability_model(x_test[:5])

