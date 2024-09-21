"""
twinc_network.py
Author: Anupama Jha <anupamaj@uw.edu>
"""

import torch
import numpy as np
from .twinc_utils import count_pos_neg, decode_chrome_order_dict, decode_list
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve


def encoder_linear_block(layer1_in,
                         layer1_out,
                         layer2_in,
                         layer2_out,
                         kernel1,
                         kernel2,
                         padding1,
                         padding2):
    """
    This function creates a sequential container using
    linear 1D convolutional layers followed by batch
    normalization.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)
    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    linear_conv = torch.nn.Sequential(first_conv,
                                      first_norm,
                                      second_conv,
                                      second_norm)
    return linear_conv


def encoder_linear_block_maxpool(layer1_in,
                                 layer1_out,
                                 layer2_in,
                                 layer2_out,
                                 pool_kernel,
                                 pool_stride,
                                 kernel1,
                                 kernel2,
                                 padding1,
                                 padding2):
    """
    This function creates a sequential container using
    max pooling followed by linear 1D convolutional layers
    and batch normalization.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param pool_kernel: int, filter size for the max pooling layer.
    :param pool_stride: int, stride size for the max pooling layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    maxpool = torch.nn.MaxPool1d(kernel_size=pool_kernel,
                                 stride=pool_stride)
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)
    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    pool_linear_conv = torch.nn.Sequential(maxpool,
                                           first_conv,
                                           first_norm,
                                           second_conv,
                                           second_norm)
    return pool_linear_conv


def encoder_nonlinear_block(layer1_in,
                            layer1_out,
                            layer2_in,
                            layer2_out,
                            kernel1,
                            kernel2,
                            padding1,
                            padding2):
    """
    This function creates a sequential container using
    linear 1D convolutional layers followed by batch
    normalization and ReLU nonlinearity.
    :param layer1_in: int, input size for the first conv1D layer.
    :param layer1_out: int, output size for the first conv1D layer.
    :param layer2_in: int, input size for the second conv1D layer.
    :param layer2_out: int, output size for the second conv1D layer.
    :param kernel1: int, filter size for the first conv1D layer.
    :param kernel2: int, filter size for the second conv1D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv1d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm1d(layer1_out)

    second_conv = torch.nn.Conv1d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)
    second_norm = torch.nn.BatchNorm1d(layer2_out)

    relu = torch.nn.ReLU(inplace=True)

    nonlinear_conv = torch.nn.Sequential(first_conv,
                                         first_norm,
                                         relu,
                                         second_conv,
                                         second_norm,
                                         relu)
    return nonlinear_conv


def decoder_linear_block(layer1_in,
                         layer1_out,
                         layer2_in,
                         layer2_out,
                         kernel1,
                         kernel2,
                         padding1,
                         padding2,
                         dilation1,
                         dilation2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers followed by batch
    normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)
    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    linear_conv = torch.nn.Sequential(first_conv,
                                      first_norm,
                                      second_conv,
                                      second_norm)
    return linear_conv


def decoder_linear_block_drop(layer1_in,
                              layer1_out,
                              layer2_in,
                              layer2_out,
                              kernel1,
                              kernel2,
                              padding1,
                              padding2,
                              dilation1,
                              dilation2,
                              drop_prob):
    """
    This function creates a sequential container using
    dropout followed by linear 2D convolutional layers and batch
    normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :param drop_prob: float, dropout probability
    :return: torch.nn.Sequential container
    """
    dropout = torch.nn.Dropout(p=drop_prob)
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)
    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    drop_linear_conv = torch.nn.Sequential(dropout,
                                           first_conv,
                                           first_norm,
                                           second_conv,
                                           second_norm)
    return drop_linear_conv


def decoder_nonlinear_block(layer1_in,
                            layer1_out,
                            layer2_in,
                            layer2_out,
                            kernel1,
                            kernel2,
                            padding1,
                            padding2,
                            dilation1,
                            dilation2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers, ReLU activation
    and batch normalization.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :param dilation1: int, dilation to add to the first layer.
    :param dilation2: int, dilation to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1,
                                 dilation=dilation1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)

    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2,
                                  dilation=dilation2)
    second_norm = torch.nn.BatchNorm2d(layer2_out)

    relu = torch.nn.ReLU(inplace=True)

    nonlinear_conv = torch.nn.Sequential(first_conv,
                                         first_norm,
                                         relu,
                                         second_conv,
                                         second_norm,
                                         relu)
    return nonlinear_conv


def decoder_final_block(layer1_in,
                        layer1_out,
                        layer2_in,
                        layer2_out,
                        kernel1,
                        kernel2,
                        padding1,
                        padding2):
    """
    This function creates a sequential container using
    linear 2D convolutional layers, batch normalization
    and ReLU activation.
    :param layer1_in: int, input size for the first conv2D layer.
    :param layer1_out: int, output size for the first conv2D layer.
    :param layer2_in: int, input size for the second conv2D layer.
    :param layer2_out: int, output size for the second conv2D layer.
    :param kernel1: (int, int), filter size for the first conv2D layer.
    :param kernel2: (int, int), filter size for the second conv2D layer.
    :param padding1: int, padding to add to the first layer.
    :param padding2: int, padding to add to the second layer.
    :return: torch.nn.Sequential container
    """
    first_conv = torch.nn.Conv2d(layer1_in,
                                 layer1_out,
                                 kernel_size=kernel1,
                                 padding=padding1)
    first_norm = torch.nn.BatchNorm2d(layer1_out)

    relu = torch.nn.ReLU(inplace=True)

    second_conv = torch.nn.Conv2d(layer2_in,
                                  layer2_out,
                                  kernel_size=kernel2,
                                  padding=padding2)

    final_conv = torch.nn.Sequential(first_conv,
                                     first_norm,
                                     relu,
                                     second_conv)
    return final_conv


class TwinCNet(torch.nn.Module):
    def __init__(self):
        """
        Constructor for the TwinCNet
        """
        super(TwinCNet, self).__init__()

        # Encoder linear layers
        self.linear_conv_encoder_1 = encoder_linear_block(layer1_in=4,
                                                          layer1_out=64,
                                                          layer2_in=64,
                                                          layer2_out=64,
                                                          kernel1=9,
                                                          kernel2=9,
                                                          padding1=4,
                                                          padding2=4)

        self.linear_conv_encoder_2 = encoder_linear_block_maxpool(layer1_in=64,
                                                                  layer1_out=96,
                                                                  layer2_in=96,
                                                                  layer2_out=96,
                                                                  pool_kernel=4,
                                                                  pool_stride=4,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_3 = encoder_linear_block_maxpool(layer1_in=96,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=5,
                                                                  pool_stride=5,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_4 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=5,
                                                                  pool_stride=5,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_5 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=5,
                                                                  pool_stride=5,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_6 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=5,
                                                                  pool_stride=5,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_7 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=2,
                                                                  pool_stride=2,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_8 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=2,
                                                                  pool_stride=2,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        self.linear_conv_encoder_9 = encoder_linear_block_maxpool(layer1_in=128,
                                                                  layer1_out=128,
                                                                  layer2_in=128,
                                                                  layer2_out=128,
                                                                  pool_kernel=2,
                                                                  pool_stride=2,
                                                                  kernel1=9,
                                                                  kernel2=9,
                                                                  padding1=4,
                                                                  padding2=4)

        # Encoder nonlinear layers
        self.nonlinear_conv_encoder_1 = encoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=64,
                                                                layer2_in=64,
                                                                layer2_out=64,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_2 = encoder_nonlinear_block(layer1_in=96,
                                                                layer1_out=96,
                                                                layer2_in=96,
                                                                layer2_out=96,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_3 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_4 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_5 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_6 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_7 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_8 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        self.nonlinear_conv_encoder_9 = encoder_nonlinear_block(layer1_in=128,
                                                                layer1_out=128,
                                                                layer2_in=128,
                                                                layer2_out=128,
                                                                kernel1=9,
                                                                kernel2=9,
                                                                padding1=4,
                                                                padding2=4)

        # Decoder linear layers
        self.linear_conv_decoder_1 = decoder_linear_block_drop(layer1_in=128,
                                                               layer1_out=32,
                                                               layer2_in=32,
                                                               layer2_out=64,
                                                               kernel1=(3, 3),
                                                               kernel2=(3, 3),
                                                               padding1=1,
                                                               padding2=1,
                                                               dilation1=1,
                                                               dilation2=1,
                                                               drop_prob=0.1)

        self.linear_conv_decoder_2 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=2,
                                                          padding2=2,
                                                          dilation1=2,
                                                          dilation2=2)

        self.linear_conv_decoder_3 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=4,
                                                          padding2=4,
                                                          dilation1=4,
                                                          dilation2=4)

        self.linear_conv_decoder_4 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=8,
                                                          padding2=8,
                                                          dilation1=8,
                                                          dilation2=8)

        self.linear_conv_decoder_5 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=16,
                                                          padding2=16,
                                                          dilation1=16,
                                                          dilation2=16)

        self.linear_conv_decoder_6 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=32,
                                                          padding2=32,
                                                          dilation1=32,
                                                          dilation2=32)

        self.linear_conv_decoder_7 = decoder_linear_block(layer1_in=64,
                                                          layer1_out=32,
                                                          layer2_in=32,
                                                          layer2_out=64,
                                                          kernel1=(3, 3),
                                                          kernel2=(3, 3),
                                                          padding1=64,
                                                          padding2=64,
                                                          dilation1=64,
                                                          dilation2=64)

        # Decoder nonlinear layers
        self.nonlinear_conv_decoder_1 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=1,
                                                                padding2=1,
                                                                dilation1=1,
                                                                dilation2=1)

        self.nonlinear_conv_decoder_2 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=2,
                                                                padding2=2,
                                                                dilation1=2,
                                                                dilation2=2)

        self.nonlinear_conv_decoder_3 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=4,
                                                                padding2=4,
                                                                dilation1=4,
                                                                dilation2=4)

        self.nonlinear_conv_decoder_4 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=8,
                                                                padding2=8,
                                                                dilation1=8,
                                                                dilation2=8)

        self.nonlinear_conv_decoder_5 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=16,
                                                                padding2=16,
                                                                dilation1=16,
                                                                dilation2=16)

        self.nonlinear_conv_decoder_6 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=32,
                                                                padding2=32,
                                                                dilation1=32,
                                                                dilation2=32)

        self.nonlinear_conv_decoder_7 = decoder_nonlinear_block(layer1_in=64,
                                                                layer1_out=32,
                                                                layer2_in=32,
                                                                layer2_out=64,
                                                                kernel1=(3, 3),
                                                                kernel2=(3, 3),
                                                                padding1=64,
                                                                padding2=64,
                                                                dilation1=64,
                                                                dilation2=64)

        # Encoder linear module
        self.linear_conv_encoder = torch.nn.ModuleList(
            [self.linear_conv_encoder_1,
             self.linear_conv_encoder_2,
             self.linear_conv_encoder_3,
             self.linear_conv_encoder_4,
             self.linear_conv_encoder_5,
             self.linear_conv_encoder_6,
             self.linear_conv_encoder_7,
             self.linear_conv_encoder_8,
             self.linear_conv_encoder_9
             ]
        )

        # Encoder nonlinear module
        self.nonlinear_conv_encoder = torch.nn.ModuleList(
            [self.nonlinear_conv_encoder_1,
             self.nonlinear_conv_encoder_2,
             self.nonlinear_conv_encoder_3,
             self.nonlinear_conv_encoder_4,
             self.nonlinear_conv_encoder_5,
             self.nonlinear_conv_encoder_6,
             self.nonlinear_conv_encoder_7,
             self.nonlinear_conv_encoder_8,
             self.nonlinear_conv_encoder_9
             ]
        )

        # Decoder linear module
        self.linear_conv_decoder = torch.nn.ModuleList(
            [self.linear_conv_decoder_1,
             self.linear_conv_decoder_2,
             self.linear_conv_decoder_3,
             self.linear_conv_decoder_4,
             self.linear_conv_decoder_5,
             self.linear_conv_decoder_6,
             self.linear_conv_decoder_7,
             self.linear_conv_decoder_2,
             self.linear_conv_decoder_3,
             self.linear_conv_decoder_4,
             self.linear_conv_decoder_5,
             self.linear_conv_decoder_6,
             self.linear_conv_decoder_7,
             self.linear_conv_decoder_2,
             self.linear_conv_decoder_3,
             self.linear_conv_decoder_4,
             self.linear_conv_decoder_5,
             self.linear_conv_decoder_6,
             self.linear_conv_decoder_7
             ]
        )

        # Decoder nonlinear module
        self.nonlinear_conv_decoder = torch.nn.ModuleList(
            [self.nonlinear_conv_decoder_1,
             self.nonlinear_conv_decoder_2,
             self.nonlinear_conv_decoder_3,
             self.nonlinear_conv_decoder_4,
             self.nonlinear_conv_decoder_5,
             self.nonlinear_conv_decoder_6,
             self.nonlinear_conv_decoder_7,
             self.nonlinear_conv_decoder_2,
             self.nonlinear_conv_decoder_3,
             self.nonlinear_conv_decoder_4,
             self.nonlinear_conv_decoder_5,
             self.nonlinear_conv_decoder_6,
             self.nonlinear_conv_decoder_7,
             self.nonlinear_conv_decoder_2,
             self.nonlinear_conv_decoder_3,
             self.nonlinear_conv_decoder_4,
             self.nonlinear_conv_decoder_5,
             self.nonlinear_conv_decoder_6,
             self.nonlinear_conv_decoder_7,
             ]
        )

        # Final layer before output
        self.final = decoder_final_block(layer1_in=64,
                                         layer1_out=5,
                                         layer2_in=5,
                                         layer2_out=1,
                                         kernel1=(1, 1),
                                         kernel2=(1, 1),
                                         padding1=0,
                                         padding2=0)

        # an output dense layer with no activation
        self.label = torch.nn.Linear(
            in_features=25,
            out_features=2
        )

        self.flatten = torch.nn.Flatten(start_dim=1)

        self.softmax = torch.nn.Softmax(dim=1)

        # Loss functions
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.cross_entropy_loss = torch.nn.BCELoss(reduction="mean")

    def forward_twin_encoder(self, x):
        """
        Both sequences pass through the same ENCODER to generate the sequence embedding.
        :param x: tensor, one-hot-encoded sequence
        :return: tensor, sequence embedding
        """
        start = True
        for linear_conv_encode, nonlinear_conv_encode in zip(self.linear_conv_encoder, self.nonlinear_conv_encoder):
            if start:
                linear_x = linear_conv_encode(x)
                x = nonlinear_conv_encode(linear_x)
                start = False
            else:
                linear_x = linear_conv_encode(x + linear_x)
                x = nonlinear_conv_encode(linear_x)
        return x

    def forward_decoder(self, x1, x2):
        """
        Takes two sequence embeddings and generates output trans Hi-C contact
        :param x1: tensor, embedding for the first sequence
        :param x2: tensor, embedding for the second sequence
        :return: tensor, output trans Hi-C contact
        """
        embed = x1[:, :, :, None] + x2[:, :, None, :]

        start = True
        for linear_conv_decode, nonlinear_conv_decode in zip(self.linear_conv_decoder, self.nonlinear_conv_decoder):
            if start:
                embed = linear_conv_decode(embed)
                embed = nonlinear_conv_decode(embed) + embed
                start = False
            else:
                lembed = linear_conv_decode(embed)
                if lembed.size() == embed.size():
                    embed = lembed + embed
                else:
                    embed = lembed
                embed = nonlinear_conv_decode(embed) + embed
        embed = self.final(embed)
        embed = torch.squeeze(embed)
        embed = self.flatten(embed)
        return embed

    def forward(self, X1, X2):
        """A forward pass of the model.
        This method takes in two nucleotide sequences X1 and X2
        and makes predictions for the trans-Hi-C contacts between
        them.
        Parameters
        ----------
        X1: torch.tensor, shape=(batch_size, 4, sequence_length)
            The one-hot encoded batch of sequences.
        X2: torch.tensor, shape=(batch_size, 4, sequence_length)
            The one-hot encoded batch of sequences.
        Returns
        -------
        y: torch.tensor, shape=(batch_size, out_length)
            The trans-Hi-C predictions.
        """
        X1 = self.forward_twin_encoder(X1)
        X2 = self.forward_twin_encoder(X2)
        X = self.forward_decoder(X1, X2)
        y = self.softmax(self.label(X))
        return y

    def fit_supervised(
            self,
            training_data,
            model_optimizer,
            validation_data,
            max_epochs=10,
            validation_iter=1000,
            device="cpu",
            best_save_model="",
            final_save_model=""
    ):
        """
        Training procedure for the supervised version of TwinCNet CNN.
        :param training_data: torch.DataLoader, training data generator
        :param model_optimizer: torch.Optimizer,  An optimizer to training our model
        :param validation_data: torch.DataLoader, validation data generator
        :param max_epochs: int, maximum epochs to run the model for
        :param validation_iter: int,After how many iterations should we compute validation stats.
        :param device: str, GPU versus CPU, defaults to CPU
        :param best_save_model: str, path to save best model
        :param final_save_model: str, path to save final model
        :return: None
        """

        best_aupr = 0
        for epoch in range(max_epochs):
            # to log cross-entropy loss to
            # average over batches
            avg_train_loss = 0
            avg_train_iter = 0
            iteration = 0
            for data in training_data:
                # Get features and label batch
                X1, X2, y = data
                # Convert them to float
                X1, X2, y = X1.float(), X2.float(), y.float()
                X1, X2, y = X1.to(device), X2.to(device), y.to(device)

                # Clear the optimizer and
                # set the model to training mode
                model_optimizer.zero_grad()
                self.train()

                # Run forward pass
                train_pred = self.forward(X1, X2)

                # Calculate the cross entropy loss
                cross_entropy_loss = self.cross_entropy_loss(train_pred, y)

                # Extract the cross entropy loss for logging
                cross_entropy_loss_item = cross_entropy_loss.item()

                # Do the back propagation
                cross_entropy_loss.backward()
                model_optimizer.step()

                # log loss to average over training batches
                avg_train_loss += cross_entropy_loss_item
                avg_train_iter += 1
                train_loss = avg_train_loss / avg_train_iter

                print(f"Epoch {epoch}, iteration {iteration},"
                      f" train loss: {train_loss:.4f},", flush=True
                      )

                # If current iteration is a
                # validation iteration
                # compute validation stats.
                if iteration % validation_iter == 0:
                    with torch.no_grad():
                        # Set the model to
                        # evaluation mode
                        self.eval()
                        y_valid = torch.empty((0, 2))
                        valid_preds = torch.empty((0, 2)).to(device)
                        cnt = 0
                        for data in validation_data:
                            # Get features and label batch
                            X1, X2, y = data
                            y_valid = torch.cat((y_valid, y))
                            # Convert them to float
                            X1, X2, y = X1.float(), X2.float(), y.float()
                            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

                            # Run forward pass
                            val_pred = self.forward(X1, X2)
                            valid_preds = torch.cat((valid_preds, val_pred))
                            valid_preds = valid_preds.to(device)
                            cnt += 1
                            if cnt > 6000:
                                break

                        count_pos_neg(np.argmax(y_valid, axis=1), set_name="validation set")
                        valid_preds, y_valid = valid_preds.to(device), y_valid.to(device)

                        # compute cross_entropy loss
                        # for the validation set.
                        cross_entropy_loss = self.cross_entropy_loss(
                            valid_preds, y_valid
                        )

                        # Extract the validation loss
                        valid_loss = cross_entropy_loss.item()

                        # Compute AUROC
                        sklearn_rocauc = roc_auc_score(
                            y_valid.cpu().numpy()[:, 1], valid_preds.cpu().numpy()[:, 1]
                        )

                        # Compute AUPR/Average precision
                        sklearn_ap = average_precision_score(
                            y_valid.cpu().numpy()[:, 1], valid_preds.cpu().numpy()[:, 1]
                        )
                        print(f"y_valid.cpu().numpy(): {y_valid.cpu().numpy()}", flush=True)
                        print(f"valid_preds.cpu().numpy(): {valid_preds.cpu().numpy()}", flush=True)

                        train_loss = avg_train_loss / avg_train_iter

                        print(
                            f"Epoch {epoch}, iteration {iteration},"
                            f" train loss: {train_loss:4.4f},"
                            f" validation loss: {valid_loss:4.4f}", flush=True
                        )

                        print(
                            f"Validation iteration {iteration}, "
                            f"AUPR: {sklearn_ap},"
                            f"AUROC: {sklearn_rocauc}", flush=True
                        )

                        if sklearn_ap > best_aupr:
                            torch.save(self.state_dict(), best_save_model)

                            best_aupr = sklearn_ap
                        avg_train_loss = 0
                        avg_train_iter = 0

                iteration += 1

        torch.save(self.state_dict(), final_save_model)


if __name__ == '__main__':
    trans_hic_model = TwinCNet()
    input_tensor1 = torch.randn(2, 4, 100000)
    input_tensor2 = torch.randn(2, 4, 100000)
    output = trans_hic_model(input_tensor1, input_tensor2)
    print(f"output: {output}, {output.shape}")
