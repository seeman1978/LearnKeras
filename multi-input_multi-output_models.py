import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

if __name__ == "__main__":
    image_input = keras.Input(shape=(32, 32, 3), name='img_input')
    timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

    x1 = layers.Conv2D(3, 3)(image_input)
    x1 = layers.GlobalMaxPooling2D()(x1)

    x2 = layers.Conv1D(3, 3)(timeseries_input)
    x2 = layers.GlobalMaxPooling1D()(x2)

    x = layers.concatenate([x1, x2])

    score_output = layers.Dense(1, name='score_output')(x)
    class_output = layers.Dense(5, name='class_output')(x)

    model = keras.Model(inputs=[image_input, timeseries_input],
                        outputs=[score_output, class_output])

    keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={'score_output': keras.losses.MeanSquaredError(),
              'class_output': keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                                  keras.metrics.MeanAbsoluteError()],
                 'class_output': [keras.metrics.CategoricalAccuracy()]},
        loss_weights={'score_output': 2., 'class_output': 1.}) #If we only passed a single loss function to the model, the same loss function would be applied to every output,

    """
    You could also chose not to compute a loss for certain outputs, if these outputs meant for prediction but not for training:
    # List loss version
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[None, keras.losses.CategoricalCrossentropy(from_logits=True)])
    
    # Or dict loss version
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={'class_output':keras.losses.CategoricalCrossentropy(from_logits=True)})
    """

    # Generate dummy Numpy data
    img_data = np.random.random_sample(size=(100, 32, 32, 3))
    ts_data = np.random.random_sample(size=(100, 20, 10))
    score_targets = np.random.random_sample(size=(100, 1))
    class_targets = np.random.random_sample(size=(100, 5))
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ({'img_input': img_data, 'ts_input': ts_data},
         {'score_output': score_targets, 'class_output': class_targets}))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    model.fit(train_dataset, epochs=3)