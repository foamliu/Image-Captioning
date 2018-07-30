from __future__ import print_function

import os

import keras
import keras.backend as K
from hyperas import optim
from hyperas.distributions import uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, CuDNNLSTM, Concatenate, Embedding, RepeatVector, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.models import Model

from config import batch_size, num_train_samples, num_valid_samples, max_token_length, vocab_size, embedding_size, \
    best_model
from data_generator import DataGenSequence


def data():
    return DataGenSequence('train'), DataGenSequence('valid')


def create_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = CuDNNLSTM(256, return_sequences=True)(x)
    text_embedding = TimeDistributed(Dense(embedding_size))(x)

    # image embedding
    image_input = Input(shape=(2048,))
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    x = Concatenate(axis=1)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM(1024, return_sequences=True, name='language_lstm_1')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM(1024, name='language_lstm_2')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples / batch_size // 10,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples / batch_size // 10)

    score, acc = model.evaluate_generator(DataGenSequence('valid'), verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(DataGenSequence('valid')))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
