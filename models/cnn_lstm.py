import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tensorflow as tf

def build_model(input_shape, num_classes=4):
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(32,7,activation="relu",padding="same")(inp)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(64,5,activation="relu",padding="same")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.LSTM(64)(x)

    x = tf.keras.layers.Dense(64,activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    out = tf.keras.layers.Dense(num_classes,activation="softmax")(x)

    return tf.keras.Model(inp,out)
