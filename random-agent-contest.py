from retro_contest.local import make

# Input is a 224 by 320 by 3 ndarray (observation)
# Action is a size-12 vector (np.zeros(12))
# Delta score is an output of the neural network
# Reward will is an output of the neural network

# Default packages
import random
import numpy as np
import tensorflow as tf

class DDQN(object):
    def __init__(self):
        self.experience_replay = [] # Experience replay will contain
        self.obs = np.zeros([224, 320, 3])
        self.delta_score = 0
        self.reward = 0
        self.state = tf.placeholder(tf.float32, [224, 320, 3], name = 'state') # Initial current_state to all zeroes

    def train(self, terminal_reward):
        minibatch = random.choices(self.experience_replay, 16)

def main():
    training_episodes = 1000
    env = make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
    obs = env.reset()
    espio = DDQN()
    espio.obs = obs
    score = 0

    # training
    for e in range(0, training_episodes):
        action = env.action_space.sample() 
        obs, rew, done, info = env.step(action)
        score_prime = info['score']
        delta_score = score_prime - score
        print(delta_score)
        espio.experience_replay.append([espio.obs, action, rew, delta_score, obs, done])
        # env.render()
        # espio.train()
        if done:
            obs = env.reset()
            self.obs = obs
            score = 0
        score = score_prime

    # finished training
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        # env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()