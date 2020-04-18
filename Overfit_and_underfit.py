import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

def get_optimizer(lr_schedule):
    return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, lr_schedule, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer(lr_schedule)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

def TinyModel(train_ds, validate_ds, lr_schedule):
    # 微模型
    tiny_model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(1)
    ])

    return compile_and_fit(train_ds, validate_ds, tiny_model, 'sizes/Tiny', lr_schedule)

def SmallModel(lr_schedule):
    # 小模型
    small_model = tf.keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return compile_and_fit(small_model, 'sizes/Small', lr_schedule)

def MediumModel(lr_schedule):
    # 中模型
    medium_model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    return compile_and_fit(medium_model, "sizes/Medium", lr_schedule)

def LargeModel(lr_schedule):
    # 大型模型
    large_model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1)
    ])
    return compile_and_fit(large_model, "sizes/large", lr_schedule)

def DrawLoss(history):
    # We can use these to plot the training and validation loss for comparison
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

if __name__ == "__main__":
    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)
    gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
    FEATURES = 28
    ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type="GZIP")

    packed_ds = ds.batch(10000).map(pack_row).unbatch()

    N_VALIDATION = int(1e3)
    N_TRAIN = int(1e4)
    BUFFER_SIZE = int(1e4)
    BATCH_SIZE = 500
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    # load data
    validate_ds = packed_ds.take(N_VALIDATION).cache()  #ensure that the loader doesn't need to re-read the data from the file on each epoch
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

    validate_ds = validate_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    #deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.
    #There is a balance between "too much capacity" and "not enough capacity".
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH*1000,
        decay_rate=1,
        staircase=False
    )

    tiny_model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(1)
    ])

    size_histories = {}
    size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny', lr_schedule)


