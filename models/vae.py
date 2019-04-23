from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


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
    epsilon = K.random_normal(shape=(batch, dim))
    return mean + sigma * epsilon


def get_vae(input_shape, latent_dim,
            filters=[32, 64, 128, 256],
            kernels=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            deconv_filters=[128, 64, 32, 3],
            deconv_kernels=[5, 5, 6, 6],
            deconv_strides=[2, 2, 2, 2],
            train=False,
            lr=0.0001,
            kl_tolerance=0.5):

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

    if train:
        # def reconstruction_loss(y_true, y_pred):
        #     return 10 * K.mean(
        #         K.square(K.flatten(y_true) - K.flatten(y_pred)), axis = -1)
        def reconstruction_loss(y_true, y_pred):
            return K.mean(
                K.sum(K.square(y_true - y_pred), axis=[1,2,3])
            )

        # def kl_loss(y_true, y_pred):
        #     return - 0.5 * K.mean(
        #         1 + K.log(sigma) - K.square(mu) - sigma, axis = -1)
        def kl_loss(y_true, y_pred):
            return K.maximum(
                - 0.5 * K.sum(1 + K.log(sigma) - K.square(mu) - sigma, axis = -1),
                kl_tolerance*latent_dim)

        
        def vae_loss(y_true, y_pred):
            return reconstruction_loss(y_true, y_pred) + kl_loss(y_true, y_pred)
        
        vae.compile(optimizer=Adam(lr=lr), loss=vae_loss, metrics=[reconstruction_loss, kl_loss])

    return vae