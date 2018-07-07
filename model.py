import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model

from config import img_rows, img_cols, num_classes, kernel


def image_embedding():
    image_model = keras.applications.resnet50(include_top=False, weights='imagenet',
                                              pooling='avg')


def build_model():
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        encoder_decoder = build_model()
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)

    parallel_model = multi_gpu_model(encoder_decoder, gpus=None)
    print(parallel_model.summary())
    plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
