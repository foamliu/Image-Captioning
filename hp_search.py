from __future__ import print_function

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, CuDNNLSTM, Concatenate, Embedding, RepeatVector, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.models import Model

from config import batch_size, num_train_samples, num_valid_samples
from config import max_token_length
from config import vocab_size, embedding_size
from data_generator import train_gen, valid_gen


def data():
    return train_gen, valid_gen


def create_model(train_generator, validation_generator):
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True)(x)
    x = Dropout({{uniform(0, 1)}})(x)
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
    if {{choice(['three', 'four'])}} == 'four':
        x = CuDNNLSTM({{choice([512, 1024, 2048])}}, return_sequences=False)(x)
    else:
        x = CuDNNLSTM(512, return_sequences=True)(x)
        x = CuDNNLSTM(512, return_sequences=False)(x)

    x = Dropout({{uniform(0, 1)}})(x)
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd', 'nadam'])}})

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=validation_generator,
        validation_steps=num_valid_samples / batch_size)

    score, acc = model.evaluate(validation_generator, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    train_generator, validation_generator = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(validation_generator))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
