import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from label_display import label_processor
import main
from data_process import train_data, train_labels, test_data, test_labels
from main import np, MAX_SEQ_LENGTH, NUM_FEATURES, os
import matplotlib.pyplot as plt

# Ensure the directory exists for saving model weights
filepath = "E:/Sreeraj/video_motion/RNN_complex_check/ckpt.weights.h5"
os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Define the Attention layer
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait)
        a = tf.expand_dims(a, axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)

def get_complex_rnn_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Use Bidirectional LSTM layers with Attention
    x = layers.Bidirectional(layers.SimpleRNN(64, return_sequences=True))(frame_features_input, mask=mask_input)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.SimpleRNN(32, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = Attention()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = Model([frame_features_input, mask_input], output)

    rnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return rnn_model

def run_complex_rnn_experiment():
    checkpoint = ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1)

    seq_model = get_complex_rnn_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=main.EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model

history, complex_rnn_model = run_complex_rnn_experiment()

def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

plot_history(history)