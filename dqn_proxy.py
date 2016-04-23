import numpy as np
import os
import tensorflow as tf
from q_network import QNetwork

class DumbKeyValue(object): pass

class DeepQNetworkProxy:
    def __init__(self, numActions, args):

        a = DumbKeyValue()
        a.agent_name = 'dq'
        a.agent_type = 'dq'

        a.conv_kernel_shapes = [
            [8,8,4,32],
            [4,4,32,64],
            [3,3,64,64]]
        a.conv_strides = [
            [1,4,4,1],
            [1,2,2,1],
            [1,1,1,1]]
        a.dense_layer_shapes = [[3136, 512]]

        a.discount_factor = .99
        a.double_dqn = False
        a.error_clipping = 1.0
        a.game = args.rom
        a.gradient_clip = 0
        a.history_length = 4
        a.learning_rate = args.learning_rate
        a.optimizer = 'rmsprop'
        a.rmsprop_decay = .95
        a.rmsprop_epsilon = .01
        a.screen_dims = [84,84]
        a.target_update_frequency = args.target_model_update_freq
        a.watch = 0

        self.qnetwork = QNetwork(a, numActions)
        self.numActions = numActions


    def inference(self, screens):
        q_values = self.qnetwork.inference(screens)
        return np.argmax(q_values)

    def train(self, batch):
        onehot = np.eye(self.numActions)
        state1 = [sample.state1.getScreens() for sample in batch]
        action = [onehot[sample.action] for sample in batch]
        reward = [sample.reward for sample in batch]
        state2 = [sample.state2.getScreens() for sample in batch]
        terminal = [sample.terminal for sample in batch]
        self.qnetwork.train(state1, action, reward, state2, terminal)
