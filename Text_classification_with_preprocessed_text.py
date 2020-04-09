import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (train_data, test_data), info = tfds.load(
        # Use the version pre-encoded with an ~8k vocabulary.
        'imdb_reviews/subwords8k',
        # Return the train/test datasets as a tuple.
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        # Return (example, label) pairs from the dataset (instead of a dictionary).
        as_supervised=True,
        # Also return the `info` structure.
        with_info=True)

    encoder = info.features['text'].encoder
    print('Vocabulary size: {}'.format(encoder.vocab_size))

    #Prepare the data for training
    BUFFER_SIZE = 1000

    train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, padded_shapes=([None], [])))

    test_batches = ( test_data.padded_batch(32, padded_shapes=([None], [])))

    #build the model
    model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1)])

    model.summary()

    #configure the model to use an optimizer and a loss function
    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    #Train the model
    history = model.fit(train_batches,
                        epochs=10,
                        validation_data=test_batches,
                        validation_steps=30)
    #Evaluate the model
    loss, accuracy = model.evaluate(test_batches)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    #We can use these to plot the training and validation loss for comparison
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()
