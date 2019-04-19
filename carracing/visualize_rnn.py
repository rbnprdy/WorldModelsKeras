"""Generates a figure of the VAE recreation on random samples"""
import argparse
import os

from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

from models.vae import get_vae
from models.rnn import get_rnn


def main(args):
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    output_file = args.output_file
    num_images = args.num_images
    offset = args.offset

    data_shape = (64, 64, 3)
    latent_dim = 32

    vae = get_vae(data_shape, latent_dim)
    vae.load_weights(checkpoint_path)
    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    rnn_train, rnn_infer = get_rnn((None, 35))
    rnn_train.load_weights('checkpoints/rnn.h5')

    filelist = os.listdir(data_dir)
    filelist.sort()
    filename = filelist[0]
    raw_data = np.load(os.path.join(data_dir, filename))
    obs = raw_data['obs'].astype(np.float) / 255.
    actions = raw_data['action']
    obs = obs[offset:offset+num_images]
    
    z_true = encoder.predict(obs)[-1]
    rnn_input = np.column_stack([z_true, actions])[:-1]
    rnn_input = np.reshape(rnn_input, (-1, 1, 35))
    z_pred = rnn_train.predict(rnn_input)

    # Sample predictions
    num_mixtures = 5
    output_sequence_width = 32
    # Flattened output sequences and mixtures
    flat = output_sequence_width*num_mixtures
    # Get number of sequences
    rollout = np.shape(z_pred)[1]
    # Extract flattened variables
    logpis_flat = z_pred[:,:,:flat]
    mus_flat = z_pred[:,:,flat:flat*2]
    sigmas_flat = z_pred[:,:,flat*2:flat*3]
    sigmas_flat.shape
    # Reshape to (batch, time, num_mixtures, output_sequence_width)
    shape = [-1, rollout, num_mixtures, output_sequence_width]
    logpis = np.reshape(logpis_flat, shape)
    mus = np.reshape(mus_flat, shape)
    sigmas = np.reshape(sigmas_flat, shape)
    # Tempearture stuff
    temperature = 0.7
    logpistemp = np.copy(logpis)/temperature
    logpistemp -= logpistemp.max()
    pistemp = np.exp(logpistemp)
    pistemp /= pistemp.sum(axis=2, keepdims=True)
    pitemp = pistemp[:,0,:,:]
    mu = mus[:,0,:,:]
    sigma = sigmas[:,0,:,:]

    def get_pi_idx(x, pdf):
        # samples from a categorial distribution
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print('error with sampling ensemble')
        return -1

    pi_idx = np.zeros((pitemp.shape[0], 32))
    chosen_mu = np.zeros((pitemp.shape[0], 32))
    chosen_sigma = np.zeros((pitemp.shape[0], 32))
    for i in range(pitemp.shape[0]):
        for j in range(32):
            idx = get_pi_idx(np.random.rand(), pitemp[i][:,j])
            pi_idx[i,j] = idx
            chosen_mu[i,j] = mu[i][idx][j]
            chosen_sigma[i,j] = sigma[i][idx][j]

    rand_gaussian = np.random.normal(size=(pitemp.shape[0], 32))*np.sqrt(temperature)
    next_z = chosen_mu+chosen_sigma*rand_gaussian
    next_z.shape

    decoder = Model(inputs=vae.get_layer('decoder').input,
                    outputs=vae.get_layer('decoder').output)

    predictions = decoder.predict(next_z)
    predictions_real = decoder.predict(z_true)

    fig, axes = plt.subplots(5, 2)
    for ax, im, gen in zip(axes, predictions_real[1:], predictions):
        ax[0].imshow(im, interpolation='nearest')
        ax[0].set_title('Actual Image')
        ax[1].imshow(gen, interpolation='nearest')
        ax[1].set_title('Decoded RNN Prediction')
    plt.savefig(output_file)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize the VAE.')
    parser.add_argument('--data_dir', '-d', default='data/')
    parser.add_argument('--checkpoint_path', '-c', default='checkpoints/vae.h5')
    parser.add_argument('--output_file', '-o', default='vae.png')
    parser.add_argument('--num_images', '-n', type=int, default=5)
    parser.add_argument('--offset', type=int, default=20)
    main(parser.parse_args())
