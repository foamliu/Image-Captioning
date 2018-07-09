import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Concatenate, Embedding, RepeatVector, Bidirectional, TimeDistributed
from keras.models import Model
from keras.utils import plot_model

from config import hidden_size, max_token_length
from config import vocab_size, embedding_size


def build_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = LSTM(256, return_sequences=True)(x)
    text_embedding = TimeDistributed(Dense(300))(x)

    # image embedding
    image_input = Input(shape=(2048,))
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    x = Concatenate(axis=1)(x)
    x = Bidirectional(LSTM(hidden_size, return_sequences=False))(x)

    output = Dense(vocab_size, activation='softmax', name='output')(x)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
