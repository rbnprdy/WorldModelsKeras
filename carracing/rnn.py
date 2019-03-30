from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
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
    return dist.prob(K.repeat_elements(y, K.shape(mus)[2], 2))

def loss_fn(#FIXME):
    return K.mean(-K.log(pdf(y_true, pis, mus, sigmas) + 1e-8), axis=(1,2))

def get_rnn(input_shape, lstm_dim=256, output_sequence_width=32, num_mixtures=5):
    inputs = Input(shape=input_shape, name='rnn_input')
    x = LSTM(lstm_dim, return_sequences=True, return_state=True, name='lstm')(inputs)
    x = Flatten(name='flatten')(x)
    pis = Dense(output_sequence_width*num_mixtures, activation='softmax', name='pis')(x)
    mus = Dense(output_sequence_width*num_mixtures, name='mus')(x)
    sigmas = Dense(output_sequence_width*num_mixtures, activation='exponential', name='sigmas')(x)
    rnn = Model(inputs, [pis_softmax, mus, sigmas_exp], name='mdn-rnn')

