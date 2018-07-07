import keras.backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, RepeatVector, LSTM, GRU, TimeDistributed, \
    Dropout, Masking, Add
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

from config import rnn_type, hidden_size
from config import vocab_size, embedding_size, max_token_length, regularizer  # ,zh_model


def build_image_embedding():
    return input, x


#
# def build_word_embedding():
#     embedding_matrix = np.zeros((vocab_size, embedding_size))
#     for index, word in tqdm(enumerate(zh_model.vocab)):
#         embedding_matrix[index, :] = zh_model[word]
#
#     input = Input(shape=[None])
#     x = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=False)(input)
#     return input, x
#

def build_model():
    # word embedding
    text_input = Input(shape=(max_token_length, vocab_size), name='text')
    x = Masking(mask_value=0.0, name='text_mask')(text_input)
    x = TimeDistributed(Dense(units=embedding_size,
                              kernel_regularizer=l2(regularizer),
                              name='text_embedding'))(x)

    text_dropout = Dropout(.5, name='text_dropout')(x)

    # image embedding
    image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    image_input = image_model.input
    for layer in image_model.layers:
        layer.trainable = False
    x = image_model.output
    x = Dense(embedding_size, activation='relu')(x)
    x = RepeatVector(max_token_length)(x)
    x = TimeDistributed(Dense(units=embedding_size,
                              kernel_regularizer=l2(regularizer),
                              name='image_embedding'))(x)
    image_dropout = Dropout(.5, name='image_dropout')(x)

    # language model
    recurrent_inputs = [text_dropout, image_dropout]
    merged_input = Add()(recurrent_inputs)
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
                                   activation='softmax'),
                             name='output')(recurrent_network)

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
