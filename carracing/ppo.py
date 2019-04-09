import numpy as np
import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from models.vae import get_vae
from models.rnn import get_rnn
from models.controller import get_controller
from env import make_env
import numba as nb
from tensorboardX import SummaryWriter

LSTM_DIM = 256
LATENT_DIM = 32

EPISODES = 100000

LOG_INTERVAL = 10

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
#EPOCHS = 10
EPOCHS = 4
#NOISE = 1.0 # Exploration noise
NOISE = 0.1

#GAMMA = 0.99
GAMMA = 0.9

REPEAT_ACTION = 8

TRIAL_SIZE = 1000
#BUFFER_SIZE = 2000
BUFFER_SIZE = 6400
BATCH_SIZE = 64
NUM_ACTIONS = 3
LR = 1e-5 # Lower lr stabilises training greatly
#LR = 1e-3

#MIN_REWARD = -0.1
MIN_REWARD = -0.2

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


def adjust_bounds(x):
    return (x + 1) / 2


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss


class Agent:
    def __init__(self):
        vae = get_vae((64, 64, 3), LATENT_DIM)
        vae.load_weights('checkpoints/vae.h5')

        self.encoder = Model(inputs=vae.input,
                        outputs=vae.get_layer('encoder').output)

        rnn_train, self.rnn = get_rnn((None, LATENT_DIM + NUM_ACTIONS), train=False)
        rnn_train.load_weights('checkpoints/rnn.h5')

        self.critic = self.build_critic()
        self.actor = self.build_actor()

        self.env = make_env('carracing', full_episode=True)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.h = np.zeros((1, LSTM_DIM))
        self.c = np.zeros((1, LSTM_DIM))
        self.observation = self.process_observation(self.env.reset(), self.env.action_space.sample())
        # self.val = False
        self.reward = []
        self.gradient_steps = 0
        self.writer = SummaryWriter('logs/')
        self.av_r = self.reward_memory()

    def process_observation(self, obs, action):
        encoded_obs = self.encoder.predict(np.expand_dims(obs, axis=0))[0]
        expanded_action = np.expand_dims(action, axis=0)
        obs_and_action = np.concatenate([encoded_obs, expanded_action], axis=1)
        self.h, self.c = self.rnn.predict([np.expand_dims(obs_and_action, axis=0), self.h, self.c])
        return np.concatenate([encoded_obs, self.h], axis=1)

    def build_actor(self):
        state_input = Input(shape=[LSTM_DIM+LATENT_DIM], name='state_input')
        advantage = Input(shape=(1,), name='advantage_input')
        old_prediction = Input(shape=(NUM_ACTIONS,), name='old_prediction_input')

        #hidden = Dense(40,
        #               activation='tanh',
        #               name='hidden')(state_input)
        hidden = state_input
        x = Dense(3,
                  activation='tanh',
                  name='output')(hidden)
        nonadjust = Lambda(lambda x: K.expand_dims(x[:, 0], axis=-1))(x)
        adjust = Lambda(lambda x: adjust_bounds(x[:, 1:]))(x)
        outputs = Concatenate()([nonadjust, adjust])

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[outputs])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):
        state_input = Input(shape=(LSTM_DIM + LATENT_DIM,), name='state_input')
        #x = Dense(40, activation='tanh')(state_input)
        x = state_input
        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.process_observation(self.env.reset(), self.env.action_space.sample())
        self.av_r = self.reward_memory()

    def get_action(self):
        p = self.actor.predict([self.observation, DUMMY_VALUE, DUMMY_ACTION])
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        return action, action_matrix, p


    # https://github.com/xtma/pytorch_car_caring/blob/master/train.py
    def step(self, action):
        total_reward = 0
        for i in range(REPEAT_ACTION):
            observation, reward, done, _ = self.env.step(action)
            if done:
                reward += 100
            if np.mean(observation[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            stop = True if self.av_r(reward) <= MIN_REWARD else False
            if done or stop:
                break
        return observation, total_reward, stop, done

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def train(self, obs, action, pred, reward):
        obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
        old_prediction = pred
        pred_values = self.critic.predict(obs[:,0,:])

        advantage = reward - pred_values
        # advantage = (advantage - advantage.mean()) / advantage.std()
        actor_loss = self.actor.fit([obs[:,0,:], advantage, old_prediction[:,0,:]], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        critic_loss = self.critic.fit([obs[:,0,:]], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        print('Actor loss:', actor_loss.history['loss'][-1], 'at', self.gradient_steps)
        print('Critic loss:', critic_loss.history['loss'][-1], 'at', self.gradient_steps)
        self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
        self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
        self.gradient_steps += 1


    def run(self):

        # self.actor.load_weights('checkpoints/actor.h5')
        # self.critic.load_weights('checkpoints/critic.h5')

        running_score = 0

        observations = []
        actions = []
        predictions = []
        self.reward = []

        while self.episode < EPISODES:

            score = 0
            self.reset_env()

            for t in range(TRIAL_SIZE):
                action, action_matrix, predicted_action = self.get_action()
                observation, reward, stop, done = self.step(action)
                self.observation = self.process_observation(observation, action)
                observations.append(self.observation)
                actions.append(action)
                predictions.append(predicted_action)
                self.reward.append(reward)
                if len(observations) == BUFFER_SIZE:
                    self.transform_reward()
                    self.train(np.array(observations), np.array(actions),
                               np.array(predictions),
                               np.reshape(np.array(self.reward), (len(self.reward), 1)))
                    observations = []
                    actions = []
                    predictions = []
                    self.reward = []
                score += reward
                if stop or done:
                    break

            running_score = running_score * 0.99 + score * 0.01

            if self.episode % LOG_INTERVAL == 0:
                self.writer.add_scalar('Score', score, self.episode)
                self.writer.add_scalar('Moving Average Score', running_score, self.episode)
                self.actor.save_weights('checkpoints/actor.h5')
                self.critic.save_weights('checkpoints/critic.h5')


    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


if __name__ == '__main__':
    ag = Agent()
    ag.run()
