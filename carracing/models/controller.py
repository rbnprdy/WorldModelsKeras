from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

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
                   activations=['tanh', 'sigmoid', 'sigmoid']):
    inputs = Input(shape=input_shape, name='input')
    # Define new layer for each neuron (so that they can have different
    # activations)
    xs = [0]*num_neurons
    for i in num_neurons:
        xs[i] = Dense(num_neurons,
                      activation=activations[i],
                      name='x{}'.format(i))(inputs)
    # Concatenate outputs
    outputs = Concatenate(name='concat')(xs)
    return Model(inputs, outputs, name='controller')