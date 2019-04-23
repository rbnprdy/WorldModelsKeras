import math

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf


def pdf(y, pis, mus, sigmas):
    """Calculates the probability density function P(Y=y | X=x) from a set
    of mixed gaussian distributions.

    # Arguments
         y - The y value for the pdf.
         pis - The mixing coefficients for the guassians.
         mus - The means for the gaussians.
         sigmas - The standard deviations for the gaussians.

    # Returns
        The pdf.
    """
    # Calculate gaussian
    result = K.exp(-1/2*K.square((y - mus) / (sigmas + 1e-8))) / \
        (K.sqrt(2*math.pi*K.square(sigmas)) + 1e-8)
    # Scale by pis
    result *= pis
    # Sum over the gaussians
    result = K.sum(result, axis=2)
    return result


def get_rnn(input_shape,
            lstm_dim=256,
            output_sequence_width=32,
            num_mixtures=5,
            train=False):
    inputs = Input(shape=input_shape, name='rnn_input')
    lstm = LSTM(lstm_dim,
                return_sequences=True,
                return_state=True,
                name='lstm')
    # Model for inference
    state_input_h = Input(shape=(lstm_dim,))
    state_input_c = Input(shape=(lstm_dim,))
    state_inputs = [state_input_h, state_input_c]
    _ , state_h, state_c = lstm(inputs,
                                initial_state=[state_input_h, state_input_c])
    inference = Model([inputs] + state_inputs,
                      [state_h, state_c],
                      name='mdn-rnn-inference')
    # Model for training
    x, _, _ = lstm(inputs)
    pis_flat = Dense(output_sequence_width*num_mixtures, name='pis')(x)
    mus_flat = Dense(output_sequence_width*num_mixtures, name='mus')(x)
    sigmas_flat = Dense(output_sequence_width*num_mixtures,
                        activation=K.exp,
                        name='sigmas')(x)
    outputs = Concatenate()([pis_flat, mus_flat, sigmas_flat])
    rnn = Model(inputs, outputs, name='mdn-rnn-training')

    if train:
        def rnn_loss(y_true, y_pred):
            # Flattened output sequences and mixtures
            flat = output_sequence_width*num_mixtures
            # Get number of sequences
            rollout = K.shape(y_pred)[1]
            # Extract flattened variables
            pis_flat = y_pred[:,:,:flat]
            mus_flat = y_pred[:,:,flat:flat*2]
            sigmas_flat = y_pred[:,:,flat*2:flat*3]
            # Reshape to (batch, time, num_mixtures, output_sequence_width)
            shape = [-1, rollout, num_mixtures, output_sequence_width]
            pis = K.reshape(pis_flat, shape)
            mus = K.reshape(mus_flat, shape)
            sigmas = K.reshape(sigmas_flat, shape)
            # Send pis through softmax
            pis = K.exp(pis) / K.sum(K.exp(pis), axis=2, keepdims=True)
            # Reshape y to be (batch, time, output_sequence_width)
            y = K.reshape(y_true, [-1, rollout, output_sequence_width])
            # Tile y to be (batch, time, num_mixtures*output_sequence_width)
            y = tf.tile(y, (1, 1, num_mixtures))
            # Reshape to (batch, time, num_mixtures, output_sequence_width)
            y = K.reshape(y, [-1, rollout, num_mixtures, output_sequence_width])
            # Pass through gaussian, then do mean of log loss.
            return K.mean(-K.log(pdf(y, pis, mus, sigmas) + 1e-8), axis=(1,2))

        rnn.compile(optimizer='adam', loss=rnn_loss)

    return rnn, inference