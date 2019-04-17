from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def adjust_bounds(x):
    return (x + 1) / 2


# CarRacing-v0 environment has 3 actions:
#   1. Steering: Real valued in [-1, 1]
#   2. Gas: Real valued in [0, 1]
#   3. Brake: Real valued in [0, 1]
#
# The output layer will have 3 neurons. The first neuron will be bounded by
# [-1, 1] using a tanh activation, and then the second two neurons will
# be bounded by [0, 1] using a sigmoid activation.
def get_controller(input_shape,
                   num_neurons=3,
                   hidden_size=None,
                   activations=['tanh', 'sigmoid', 'sigmoid']):
    inputs = Input(shape=input_shape, name='input')
    # Define new layer for each neuron (so that they can have different
    # activations)
    if hidden_size:
        hidden = Dense(hidden_size,
                       activation='tanh',
                       name='hidden')(inputs)
    else:
        hidden = inputs

    x = Dense(num_neurons,
              activation='tanh',
              name='output')(hidden)
    nonadjust = Lambda(lambda x: K.expand_dims(x[:, 0], axis=-1))(x)
    adjust = Lambda(lambda x: adjust_bounds(x[:, 1:]))(x)
    outputs = Concatenate()([nonadjust, adjust])

    #xs = [0]*num_neurons
    #for i in range(num_neurons):
    #    xs[i] = Dense(1,
    #                  kernel_initializer='lecun_uniform',
    #                  activation=activations[i],
    #                  name='x{}'.format(i))(hidden)
    # Concatenate outputs
    #outputs = Concatenate(name='concat')(xs)
    return Model(inputs, outputs, name='controller')
