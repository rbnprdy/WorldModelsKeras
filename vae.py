from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

# Taken from https://keras.io/examples/variational_autoencoder/
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_vae(input_shape, latent_dim, filters=16, kernel_size=4):

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
    
    shape = K.int_shape(x)
    
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    mu = Dense(latent_dim, name='z_mean')(x)
    sigma = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, sigma])

    encoder = Model(inputs, [mu, sigma, z], name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2
    outputs = Conv2DTranspose(filters=input_shape[-1],
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0]

    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae
