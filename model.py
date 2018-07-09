import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, GRU, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

from config import rnn_type, hidden_size
from config import vocab_size, embedding_size, regularizer
from utils import load_word_embedding


def build_model():
    embedding_matrix = load_word_embedding()
    # word embedding
    text_input = Input(shape=[None])
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],
                               trainable=False)(text_input)

    # image embedding
    image_input = Input(shape=(2048,))
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    recurrent_inputs = [image_embedding, text_embedding]
    merged_input = Concatenate(axis=1)(recurrent_inputs)
    if rnn_type == 'lstm':
        recurrent_network = LSTM(hidden_size, return_sequences=False, name='recurrent_network')(merged_input)

    elif rnn_type == 'gru':
        recurrent_network = GRU(hidden_size, return_sequences=False, name='recurrent_network')(merged_input)
    else:
        raise Exception('Invalid rnn type')

    output = Dense(vocab_size, activation='linear', name='output')(recurrent_network)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
