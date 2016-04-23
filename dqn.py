import state
import numpy as np
import random
import os
import tensorflow as tf

gamma = .99

class GradientClippingOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="GradientClipper"):
        super(GradientClippingOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = self.optimizer.compute_gradients(*args, **kwargs)
        clipped_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is not None:
                clipped_grads_and_vars.append((tf.clip_by_value(grad, -1, 1), var))
            else:
                clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars

    def apply_gradients(self, *args, **kwargs):
        return self.optimizer.apply_gradients(*args, **kwargs)

class DeepQNetwork:
    def __init__(self, numActions, baseDir, args):
        
        self.numActions = numActions
        self.baseDir = baseDir
        self.saveModelFrequency = args.save_model_freq
        self.targetModelUpdateFrequency = args.target_model_update_freq
        
        self.actionCount = 0
        self.batchCount = 0
        # 250k environment steps is the same as 1e6 game frames
        self.annealingPeriod = 250000 if args.model is None else 0
        self.staleSess = None

        
        tf.set_random_seed(123456)
        
        with tf.device(None):
          self.sess = tf.Session()

          # First layer takes a screen, and shrinks by 2x
          self.x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4])
          print('x %s %s' % (self.x.get_shape(), self.x.dtype))

          x_normalized = tf.to_float(self.x) / 255.0
          print('x_normalized %s %s' % (x_normalized.get_shape(), x_normalized.dtype))

          # Second layer convolves 32 8x8 filters with stride 4 with relu
          W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
          b_conv1 = tf.Variable(tf.fill([32], 0.1))
          
          h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
          print('h_conv1 %s' % (h_conv1.get_shape()))
          
          # Third layer convolves 64 4x4 filters with stride 2 with relu
          W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
          b_conv2 = tf.Variable(tf.fill([64], 0.1))
          
          h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
          print('h_conv2 %s' % (h_conv2.get_shape()))
          
          # Fourth layer convolves 64 3x3 filters with stride 1 with relu
          W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
          b_conv3 = tf.Variable(tf.fill([64], 0.1))
          
          h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
          print('h_conv3 %s' % (h_conv3.get_shape()))
          
          # Fifth layer is fully connected with 512 relu units
          W_fc1 = tf.Variable(tf.truncated_normal([11 * 11 * 64, 512], stddev=0.01))
          b_fc1 = tf.Variable(tf.fill([512], 0.1))
          
          h_conv3_flat = tf.reshape(h_conv3, [-1, 11 * 11 * 64])
          print('h_conv3_flat %s' % (h_conv3_flat.get_shape()))
          
          h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
          print('h_fc1 %s' % (h_fc1.get_shape()))
          
          # Sixth (Output) layer is fully connected linear layer
          W_fc2 = tf.Variable(tf.truncated_normal([512, numActions], stddev=0.01))
          b_fc2 = tf.Variable(tf.fill([numActions], 0.1))
          
          self.y = tf.matmul(h_fc1, W_fc2) + b_fc2
          print('y %s' % (self.y.get_shape()))
          self.best_action = tf.argmax(self.y, 1)

          self.a = tf.placeholder(tf.float32, shape=[None, numActions])
          print('a %s' % (self.a.get_shape()))
          self.y_ = tf.placeholder(tf.float32, [None])
          print('y_ %s' % (self.y_.get_shape()))
          
          self.y_a = tf.reduce_sum(tf.mul(self.y, self.a), reduction_indices=1)
          print('y_a %s' % (self.y_a.get_shape()))
          
          difference = tf.abs(self.y_a - self.y_)
          quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
          linear_part = difference - quadratic_part
          errors = (0.5 * tf.square(quadratic_part)) + linear_part
          self.loss = tf.reduce_sum(errors)
          #self.loss = tf.reduce_mean(tf.square(self.y_a - self.y_))

          # (??) learning rate
          # Note tried gradient clipping with rmsprop with this particular loss function and it seemed to suck
          # Perhaps I didn't run it long enough
          #optimizer = GradientClippingOptimizer(tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01))
          optimizer = tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01)
          self.train_step = optimizer.minimize(self.loss)

          self.saver = tf.train.Saver(max_to_keep=25)

          # Initialize variables
          self.sess.run(tf.initialize_all_variables())

          if args.model is not None:
              print('Loading from model file %s' % (args.model))
              self.saver.restore(self.sess, args.model)

    def chooseAction(self, state, overrideEpsilon=None):
        self.actionCount += 1
        
        futureReward = 0
        
        # e-greedy selection
        # Per dqn paper we anneal epsilon from 1 to .1 over the first 1e6 frames and
        # then .1 thereafter (??)
        if overrideEpsilon is not None:
            epsilon = overrideEpsilon
        else:
            epsilon = (1.0 - 0.9 * self.actionCount / self.annealingPeriod) if self.actionCount < self.annealingPeriod else .1

        if random.random() > (1 - epsilon):
            nextAction = random.randrange(self.numActions)
        else:
            screens = np.reshape(state.getScreens(), (1, 84, 84, 4))
            best_action_tensor, y_tensor = self.sess.run([self.best_action, self.y], {self.x: screens})
            #best_action_tensor =  self.best_action.eval(feed_dict={self.x: screens})
            nextAction = best_action_tensor[0]
            futureReward = y_tensor[0, nextAction]

        return nextAction, futureReward
        
    def train(self, batch):
        
        self.batchCount += 1 # Increment first so we don't save the model on the first run through

        # Use a stale session to evaluate to improve stability per nature paper (I dont deeply understand this (??))
        evalSess = self.sess if self.staleSess is None else self.staleSess

        x2 = [b.state2.getScreens() for b in batch]
        y2 = self.y.eval(feed_dict={self.x: x2}, session=evalSess)

        x = [b.state1.getScreens() for b in batch]
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
        
        if self.batchCount % self.targetModelUpdateFrequency == 0 or self.batchCount % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=self.batchCount)
            
            if self.batchCount % self.targetModelUpdateFrequency == 0:
                if self.staleSess is not None:
                    self.staleSess.close()
                self.staleSess = tf.Session()
                self.saver.restore(self.staleSess, savedPath)
