from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras import backend as K


def sampling(args):
    """Samples from a gaussian.

    # Arguments
        args (tensor): mean and standard deviation of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    mean, sigma = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return mean + sigma * epsilon


def get_vae(input_shape, latent_dim,
            filters=[32, 64, 128, 256],
            kernels=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            deconv_filters=[128, 64, 32, 3],
            deconv_kernels=[5, 5, 6, 6],
            deconv_strides=[2, 2, 2, 2]):

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for f, k, s in zip(filters, kernels, strides):
        x = Conv2D(filters=f,
                   kernel_size=k,
                   activation='relu',
                   strides=s)(x)

    shape = K.int_shape(x)

    x = Flatten()(x)
    mu = Dense(latent_dim, name='mean')(x)
    sigma = Dense(latent_dim, activation=K.exp, name='sigma')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, sigma])

    encoder = Model(inputs, [mu, sigma, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(filters[-1]*kernels[-1], name='decoder_dense')(latent_inputs)
    x = Reshape((1, 1, filters[-1]*kernels[-1]))(x)
    for f, k, s in zip(deconv_filters[:-1],
                       deconv_kernels[:-1],
                       deconv_strides[:-1]):
        x = Conv2DTranspose(filters=f,
                            kernel_size=k,
                            strides=s,
                            activation='relu')(x)

    outputs = Conv2DTranspose(filters=deconv_filters[-1],
                              kernel_size=deconv_kernels[-1],
                              strides=deconv_strides[-1],
                              activation='sigmoid',
                              name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0]*input_shape[1]

    kl_loss = 1 + K.log(sigma) - K.square(mu) - sigma
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    return vae
