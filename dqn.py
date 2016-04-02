import state
import numpy as np
import random
import os
import tensorflow as tf

# (??)
gamma = .99

class DeepQNetwork:
    def __init__(self, width, height, numActions, baseDir, learningRate, modelFile, saveModelFrequency, evalModelUpdateFrequency):
        self.numActions = numActions
        self.width = width
        self.height = height
        self.baseDir = baseDir
        self.saveModelFrequency = saveModelFrequency
        self.evalModelUpdateFrequency = evalModelUpdateFrequency
        self.actionCount = 0
        self.lastAction = 0
        self.lastActionFutureReward = 0
        self.batchCount = 0
        self.annealingPeriod = 1e6 if modelFile is None else 0
        self.staleSess = None

        
        tf.set_random_seed(123456)
        
        with tf.device(None):
          self.sess = tf.Session()

          # First layer takes a screen, and shrinks by 2x
          self.x = tf.placeholder(tf.uint8, shape=[None, 105, 80, 4])
          print('x %s' % (self.x.get_shape()))

          # Second layer convolves 32 8x8 filters with stride 4 with relu
          W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1))
          b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
          
          h_conv1 = tf.nn.relu(tf.nn.conv2d(tf.to_float(self.x), W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
          print('h_conv1 %s' % (h_conv1.get_shape()))
          
          # Third layer convolves 64 4x4 filters with stride 2 with relu
          W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
          b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
          
          h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
          print('h_conv2 %s' % (h_conv2.get_shape()))
          
          # Fourth layer convolves 64 3x3 filters with stride 1 with relu
          W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
          b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]))
          
          h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
          print('h_conv3 %s' % (h_conv3.get_shape()))
          
          # Fifth layer is fully connected with 512 relu units
          W_fc1 = tf.Variable(tf.truncated_normal([14 * 10 * 64, 512], stddev=0.1))
          b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
          
          h_conv2_flat = tf.reshape(h_conv2, [-1, 14 * 10 * 64])
          print('h_conv2_flat %s' % (h_conv2_flat.get_shape()))
          
          h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
          print('h_fc1 %s' % (h_fc1.get_shape()))
          
          # Sixth (Output) layer is fully connected linear layer
          W_fc2 = tf.Variable(tf.truncated_normal([512, numActions], stddev=0.1))
          b_fc2 = tf.Variable(tf.constant(0.1, shape=[numActions]))
          
          self.y = tf.matmul(h_fc1, W_fc2) + b_fc2
          print('y %s' % (self.y.get_shape()))
          self.best_action = tf.argmax(self.y, 1)

          self.a = tf.placeholder(tf.float32, shape=[None, numActions])
          print('a %s' % (self.a.get_shape()))
          self.y_ = tf.placeholder(tf.float32, [None])
          print('y_ %s' % (self.y_.get_shape()))
          
          self.y_a = tf.reduce_sum(tf.mul(self.y, self.a), reduction_indices=1)
          print('y_a %s' % (self.y_a.get_shape()))
          
          clipped_error = tf.clip_by_value(self.y_a - self.y_, -1, 1) # Per nature paper.  Am I getting this right?
          self.loss = tf.reduce_mean(tf.square(clipped_error))

          # (??) learning rate
          self.train_step = tf.train.RMSPropOptimizer(learningRate, decay=.95, epsilon=.01).minimize(self.loss)

          self.saver = tf.train.Saver(max_to_keep=25)

          # Initialize variables
          self.sess.run(tf.initialize_all_variables())

          if modelFile is not None:
              print('Loading from model file %s' % (modelFile))
              self.saver.restore(self.sess, modelFile)

    def chooseAction(self, state):
        self.actionCount += 1
        
        futureReward = 0
        
        # Only select actions every 4th frame per dqn paper (??)
        
        if self.actionCount % 4 == 0:
            # e-greedy selection
            # Per dqn paper we anneal epsilon from 1 to .1 over the first 1e6 frames and
            # then .1 thereafter (??)
            epsilon = (1.0 - 0.9 * self.actionCount / self.annealingPeriod) if self.actionCount < self.annealingPeriod else .1
            if random.random() > (1 - epsilon):
                nextAction = random.randrange(self.numActions)
            else:
                screens = np.reshape(state.screens, (1, 105, 80, 4))
                best_action_tensor, y_tensor = self.sess.run([self.best_action, self.y], {self.x: screens})
                #best_action_tensor =  self.best_action.eval(feed_dict={self.x: screens})
                nextAction = best_action_tensor[0]
                futureReward = y_tensor[0, nextAction]
        else:
            nextAction = self.lastAction
            futureReward = self.lastActionFutureReward

        self.lastAction = nextAction   
        self.lastActionFutureReward = futureReward   
        return nextAction, futureReward
        
    def train(self, batch):
        
        self.batchCount += 1 # Increment first so we don't save the model on the first run through

        # Use a stale session to evaluate to improve stability per nature paper (I dont deeply understand this (??))
        evalSess = self.sess if self.staleSess is None else self.staleSess

        x2 = [b.state2.screens for b in batch]
        y2 = self.y.eval(feed_dict={self.x: x2}, session=evalSess)

        x = [b.state1.screens for b in batch]
        a = np.zeros((len(batch), self.numActions))
        y_ = np.zeros(len(batch))
        
        for i in range(0, len(batch)):
            a[i, batch[i].action] = 1
            if batch[i].terminal:
                y_[i] = batch[i].reward
            else:
                y_[i] = batch[i].reward + gamma * np.max(y2[i])

        self.train_step.run(feed_dict={
            self.x: x,
            self.a: a,
            self.y_: y_
        }, session=self.sess)
        
        if self.batchCount % self.evalModelUpdateFrequency == 0 or self.batchCount % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=self.batchCount)
            
            if self.batchCount % self.evalModelUpdateFrequency == 0:
                if self.staleSess is not None:
                    self.staleSess.close()
                self.staleSess = tf.Session()
                self.saver.restore(self.staleSess, savedPath)
