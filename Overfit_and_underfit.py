import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from matplotlib import pyplot as plt
import pathlib
import shutil

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

def compile_and_fit(model, name, lr_schedule, optimizer=None, max_epochs=1000):
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
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    validation_steps=100,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

def TinyModel(lr_schedule):
    # 微模型
    tiny_model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
        layers.Dense(1)
    ])

    return compile_and_fit(tiny_model, 'sizes/Tiny', lr_schedule)

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

def LargeL2Model(lr_schedule):
    # 大型模型
    large_model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    return compile_and_fit(large_model, "regularizers/l2", lr_schedule)

if __name__ == "__main__":
    logdir = pathlib.Path() / "tensorboard_logs"
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

    validate_ds = validate_ds.repeat().batch(BATCH_SIZE)
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

    #train
    size_histories = {}
    size_histories['Tiny'] = TinyModel(lr_schedule)
    #size_histories['Small'] = SmallModel(lr_schedule)
    #size_histories['Medium'] = MediumModel(lr_schedule)
    #size_histories['large'] = LargeModel(lr_schedule)

    #画出损失
    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(size_histories)
    a = plt.xscale('log')
    plt.xlim([5, max(plt.xlim())])
    plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs [Log Scale]")
    plt.show()

    #Strategies to prevent overfitting
    shutil.rmtree(logdir / 'regularizers/Tiny', ignore_errors=True)
    shutil.copytree(logdir / 'sizes/Tiny', logdir / 'regularizers/Tiny')

    regularizer_histories = {}
    regularizer_histories['Tiny'] = size_histories['Tiny']
    regularizer_histories['l2'] = LargeL2Model(lr_schedule)

    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])
    plt.show()