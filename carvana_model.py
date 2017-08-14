from keras.models import Model
from keras.layers import Input, concatenate, MaxPooling2D, UpSampling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras import backend as K
import image_ops

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def dice_coeff(y_true, y_pred):
    """
    Calculates Dice coefficient
    :param y_true: ground truth - train masks in our case
    :param y_pred: predictions - predicted masks in our case
    """
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersect = K.sum(y_pred_f * y_true_f)
    y_sum = K.sum(y_pred_f) + K.sum(y_true_f)
    return (2.0 * intersect + 1.0) / (y_sum + 1.0)


def double_conv_layer(inputs, filter_size, dropout, batch_norm, data_format='channels_last'):
    """
    Creates double convolution layer for U-Net model
    :param inputs: inputs for the layer
    :param filter_size: filter size of the convolutional layer
    :param dropout: dropout value after the convolution
    :param batch_norm: True for using batch normalization, False - otherwise
    :return: double convolution layer as described in the paper
    """
    conv = Conv2D(
        filters=filter_size,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=data_format)(inputs)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)

    conv = Activation('relu')(conv)
    conv = Conv2D(
        filter_size,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=data_format)(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)

    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def UNET_Carvana(dropout_val=0.05, batch_norm=True):
    """
    Creates U-Net model (https://arxiv.org/abs/1505.04597)
    :param dropout: dropout value after the convolution
    :param batch_norm: True for using batch normalization, False - otherwise
    :return: U-Net model as described in the paper
    """
    inputs = Input(shape=(image_ops.IMAGE_HEIGHT_TRAIN, image_ops.IMAGE_WIDTH_TRAIN, INPUT_CHANNELS))
    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    conv7 = double_conv_layer(up6, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    conv12 = Conv2D(
        OUTPUT_MASK_CHANNELS,
        kernel_size=(1, 1),
        strides=(1, 1))(conv11)

    if batch_norm:
        conv12 = BatchNormalization(axis=3)(conv12)

    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=inputs, outputs=conv12)
    return model
