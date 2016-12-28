"""
### NOTICE ###

You need to upload this file.
You can add any function you want in this file.

"""
import argparse
import random
import sys

# import cv2
import numpy as np

from dqn_atari import dqn
from dqn_atari.state import State


class Agent(object):
    def __init__(self, sess, min_action_set):
        self.sess = sess
        self.min_action_set = min_action_set
        self.build_dqn()
        self.state = State()
        # cv2.startWindowThread()
        # cv2.namedWindow('Breakout')

    def build_dqn(self):
        """
        # TODO
            You need to build your DQN here.
            And load the pre-trained model named as './best_model.ckpt'.
            For example, 
                saver.restore(self.sess, './best_model.ckpt')
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--compress-replay", action='store_true', help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
        parser.add_argument("--normalize-weights", action='store_true', default=True, help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
        parser.add_argument("--save-model-freq", type=int, default=10000, help="save the model once per 10000 training sessions")
        parser.add_argument("--learning-rate", type=float, default=0.00025, help="learning rate (step size for optimization algo)")
        parser.add_argument("--target-model-update-freq", type=int, default=10000, help="how often (in steps) to update the target model.  Note nature paper says this is in 'number of parameter updates' but their code says steps. see tinyurl.com/hokp4y8")
        parser.add_argument("--model", default=sys.path[0] + '/best_model.ckpt', help="tensorflow model checkpoint file to initialize from")
        args = parser.parse_args([])

        self.dqn = dqn.DeepQNetwork(4, '/tmp', args)

    def getSetting(self):
        """
        # TODO
            You can only modify these three parameters.
            Adding any other parameters are not allowed.
            1. action_repeat: number of time for repeating the same action 
            2. screen_type: return 0 for RGB; return 1 for GrayScale
        """
        action_repeat = 4
        screen_type = 0
        return action_repeat, screen_type

    def play(self, screen):
        """
        # TODO
            The "action" is your DQN argmax ouput.
            The "min_action_set" is used to transform DQN argmax ouput into real action number.
            For example,
                 DQN output = [0.1, 0.2, 0.1, 0.6]
                 argmax = 3
                 min_action_set = [0, 1, 3, 4]
                 real action number = 4
        """
        # cv2.imshow('Breakout', screen)
        self.state = self.state.stateByAddingScreen(screen, 0)
        screens = np.reshape(self.state.getScreens(), (1, 84, 84, 4))
        if random.random() < 0.01:
            action = 1
        else:
            action = self.dqn.inference(screens)

        return self.min_action_set[action]
