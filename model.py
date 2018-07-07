import keras.backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, RepeatVector, Embedding, Concatenate, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.utils import plot_model

from config import image_embedding_size, word_embedding_size, vocab_size
from config import rnn_type, bidirectional_rnn, rnn_layers, rnn_output_size, rnn_dropout_rate


def build_image_embedding():
    image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    input = image_model.input
    for layer in image_model.layers:
        layer.trainable = False
    x = image_model.output
    x = Dense(image_embedding_size, activation='relu')(x)
    output = RepeatVector(1)(x)
    return input, output


def build_word_embedding():
    input = Input(shape=[None])
    output = Embedding(input_dim=vocab_size, output_dim=word_embedding_size)(input)
    return input, output


def build_sequence_model(sequence_input):
    RNN = GRU if rnn_type == 'gru' else LSTM

    def rnn():
        rnn = RNN(units=rnn_output_size,
                  return_sequences=True,
                  dropout=rnn_dropout_rate,
                  recurrent_dropout=rnn_dropout_rate,
                  implementation=2)
        rnn = Bidirectional(rnn) if bidirectional_rnn else rnn
        return rnn

    x = sequence_input
    for _ in range(rnn_layers):
        rnn_out = rnn()(x)
        x = rnn_out
    x = TimeDistributed(Dense(vocab_size))(x)

    return x


def build_model():
    image_input, image_embedding = build_image_embedding()
    sentence_input, word_embedding = build_word_embedding()
    sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])
    sequence_output = build_sequence_model(sequence_input)

    model = Model(inputs=[image_input, sentence_input], outputs=sequence_output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
