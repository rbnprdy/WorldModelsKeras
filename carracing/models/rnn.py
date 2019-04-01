from tensorflow.keras.layers import Input, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfd

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
    dist = tfd.Normal(loc=mus, scale=sigmas)
    return dist.prob(K.repeat_elements(y, K.shape(mus)[2], 2))*pi


def rnn_loss(y_true, y_pred):
    pis, mus, sigmas = y_pred[0], y_pred[1], y_pred[2]
    return K.mean(-K.log(pdf(y_true, pis, mus, sigmas) + 1e-8), axis=(1,2))


def get_rnn(input_shape, lstm_dim=256, output_sequence_width=32, num_mixtures=5):
    inputs = Input(shape=input_shape, name='rnn_input')
    x = LSTM(lstm_dim, return_sequences=True, return_state=True, name='lstm')(inputs)
    pis = Dense(output_sequence_width*num_mixtures, activation='softmax', name='pis')(x[0])
    mus = Dense(output_sequence_width*num_mixtures, name='mus')(x[0])
    sigmas = Dense(output_sequence_width*num_mixtures, activation=K.exp, name='sigmas')(x[0])
    rnn = Model(inputs, [pis, mus, sigmas], name='mdn-rnn')
    rnn.summary()

    return rnn
