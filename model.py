import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, CuDNNGRU, Concatenate, Embedding, RepeatVector, TimeDistributed, Dropout
from keras.models import Model
from keras.utils import plot_model

from config import max_token_length
from config import vocab_size, embedding_size


def build_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = CuDNNGRU(512, return_sequences=True)(x)
    x = Dropout(.5)(x)
    text_embedding = TimeDistributed(Dense(embedding_size))(x)

    # image embedding
    image_input = Input(shape=(2048,))
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    x = Concatenate(axis=1)(x)
    x = CuDNNGRU(512, return_sequences=False)(x)
    x = Dropout(.5)(x)
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
