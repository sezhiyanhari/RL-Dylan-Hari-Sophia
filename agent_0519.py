from retro_contest.local import make

# Input is a 224 by 320 by 3 ndarray (observation)
# Action is a size-12 vector (np.zeros(12))
# Delta score is an output of the neural network
# Reward will is an output of the neural network

# Default packages
import random
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
batch_size = 16

class DDQN(object):
    def __init__(self):
        self.experience_replay = np.zeros((0,6)) # Experience replay will contain
        self.state = tf.placeholder(tf.float32, [None, 215], name = 'state') # Initial current_state to all zeroes
        self.state_prime = tf.placeholder(tf.float32, [None, 215], name = 'state_prime')

        xavier =  tf.contrib.layers.xavier_initializer(uniform = True, seed = None, dtype = tf.float32)

        self.dropped_state = tf.layers.dropout(self.state, rate = 0.2)
        self.dropped_state_prime = tf.layers.dropout(self.state_prime, rate = 0.2)
        
        # Neural Network
        with tf.name_scope('Neural_network') as scope:
            with tf.name_scope('hidden_layer') as scope:
                self.W1 = tf.Variable(xavier([215, 4096]))
                self.b1 = tf.Variable(xavier([4096]))
                self.h1 = tf.tanh(tf.matmul(self.dropped_state, self.W1) + self.b1) # None by 4096 matrix

        self.true_rewards = tf.placeholder(tf.float32, [None], name = 'true_reward')
        self.q_rewards = tf.reduce_mean(self.h1)
        self.compressed_q = tf.reshape(self.q_rewards, [-1])
        self.loss = tf.losses.mean_squared_error(self.compressed_q, self.true_rewards)
        self.train_step = tf.train.RMSPropOptimizer(0.00020, momentum=0.95, use_locking=False, centered=False, name='RMSProp').minimize(self.loss)

        tf.global_variables_initializer().run()
    
    def train(self):
        minibatch = self.experience_replay[np.random.choice(self.experience_replay.shape[0], batch_size), :]
        true_rewards = minibatch[:, 2] # scalar values
        done = minibatch[:, 5] # vectors of size 12
        state = np.concatenate(minibatch[:, 0]).reshape((batch_size, -1))
        print(state.shape)
        """summary, _ = sess.run([self.train_step], { self.state : board,
                                                   self.state_prime: })
        """

def main():
    training_episodes = 1000
    env = make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
    obs = env.reset()
    espio = DDQN()
    score = 0

    # training
    for e in range(0, training_episodes):
        action = env.action_space.sample() 
        obs_prime, rew, done, info = env.step(action)
        score_prime = info['score']
        delta_score = score_prime - score
        
        D = (obs_prime.flatten(), action, rew, delta_score, obs.flatten(), done)
        espio.experience_replay = np.append(espio.experience_replay, [D], axis = 0)
        # env.render()
        # espio.train()
        obs = obs_prime
        espio.train()
        score = score_prime # reset score to score_prime
        if done:
            obs = env.reset()
            score = 0

    obs = env.reset() # reset before testing begins

    # finished training
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        # env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
