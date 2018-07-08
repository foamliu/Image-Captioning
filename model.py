import keras.backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, LSTM, GRU, TimeDistributed, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

from config import rnn_type, hidden_size
from config import vocab_size, embedding_size, regularizer, embedding_matrix


def build_model():
    # word embedding
    text_input = Input(shape=[None])
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix],
                               trainable=False)(text_input)

    # image embedding
    image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    image_input = image_model.input
    for layer in image_model.layers:
        layer.trainable = False
    x = image_model.output
    x = Dense(embedding_size, activation='relu')(x)
    # the image I is only input once
    x = RepeatVector(1)(x)
    image_embedding = Dense(units=embedding_size,
                            kernel_regularizer=l2(regularizer),
                            name='image_embedding')(x)

    # language model
    recurrent_inputs = [image_embedding, text_embedding]
    merged_input = Concatenate(axis=1)(recurrent_inputs)
    if rnn_type == 'lstm':
        recurrent_network = LSTM(units=hidden_size,
                                 recurrent_regularizer=l2(regularizer),
                                 kernel_regularizer=l2(regularizer),
                                 bias_regularizer=l2(regularizer),
                                 return_sequences=True,
                                 name='recurrent_network')(merged_input)

    elif rnn_type == 'gru':
        recurrent_network = GRU(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network')(merged_input)
    else:
        raise Exception('Invalid rnn type')

    output = TimeDistributed(Dense(units=vocab_size,
                                   kernel_regularizer=l2(regularizer),
                                   activation='linear'),
                             name='output')(recurrent_network)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
