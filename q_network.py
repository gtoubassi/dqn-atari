import tensorflow as tf
import os
import numpy as np


class QNetwork():

	def __init__(self, args, num_actions):
		''' Build tensorflow graph for deep q network '''

		print("Initializing Q-Network")

		self.discount_factor = args.discount_factor
		self.target_update_frequency = args.target_update_frequency
		self.total_updates = 0
		self.path = '../saved_models/' + args.game + '/' + args.agent_type + '/' + args.agent_name
		if not os.path.exists(self.path):
   			os.makedirs(self.path)
		self.name = args.agent_name

		# input placeholders
		self.observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="observation")
		self.actions = tf.placeholder(tf.float32, shape=[None, num_actions], name="actions") # one-hot matrix because tf.gather() doesn't support multidimensional indexing yet
		self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
		self.next_observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="next_observation")
		self.terminals = tf.placeholder(tf.float32, shape=[None], name="terminals")
		self.normalized_observation = self.observation / 255.0
		self.normalized_next_observation = self.next_observation / 255.0

		num_conv_layers = len(args.conv_kernel_shapes)
		assert(num_conv_layers == len(args.conv_strides))
		num_dense_layers = len(args.dense_layer_shapes)

		last_policy_layer = None
		last_target_layer = None
		self.update_target = []
		self.policy_network_params = []
		self.param_names = []

		# initialize convolutional layers
		for layer in range(num_conv_layers):
			policy_input = None
			target_input = None
			if layer == 0:
				policy_input = self.normalized_observation
				target_input = self.normalized_next_observation
			else:
				policy_input = last_policy_layer
				target_input = last_target_layer

			last_layers = self.conv_relu(policy_input, target_input, 
				args.conv_kernel_shapes[layer], args.conv_strides[layer], layer)
			last_policy_layer = last_layers[0]
			last_target_layer = last_layers[1]

		# initialize fully-connected layers
		for layer in range(num_dense_layers):
			policy_input = None
			target_input = None
			if layer == 0:
				input_size = args.dense_layer_shapes[0][0]
				policy_input = tf.reshape(last_policy_layer, shape=[-1, input_size])
				target_input = tf.reshape(last_target_layer, shape=[-1, input_size])
			else:
				policy_input = last_policy_layer
				target_input = last_target_layer

			last_layers = self.dense_relu(policy_input, target_input, args.dense_layer_shapes[layer], layer)
			last_policy_layer = last_layers[0]
			last_target_layer = last_layers[1]


		# initialize q_layer
		last_layers = self.dense_linear(
			last_policy_layer, last_target_layer, [args.dense_layer_shapes[-1][-1], num_actions])
		self.policy_q_layer = last_layers[0]
		self.target_q_layer = last_layers[1]

		self.loss = self.build_loss(args.error_clipping, num_actions, args.double_dqn)

		if (args.optimizer == 'rmsprop') and (args.gradient_clip <= 0):
			self.train_op = tf.train.RMSPropOptimizer(
				args.learning_rate, decay=args.rmsprop_decay, momentum=0.0, epsilon=args.rmsprop_epsilon).minimize(self.loss)
		elif (args.optimizer == 'graves_rmsprop') or (args.optimizer == 'rmsprop' and args.gradient_clip > 0):
			self.train_op = self.build_rmsprop_optimizer(args.learning_rate, args.rmsprop_decay, args.rmsprop_epsilon, args.gradient_clip, args.optimizer)

		self.saver = tf.train.Saver(self.policy_network_params)

		if not args.watch:
			param_hists = [tf.histogram_summary(name, param) for name, param in zip(self.param_names, self.policy_network_params)]
			self.param_summaries = tf.merge_summary(param_hists)

		# start tf session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33333)  # avoid using all vram for GTX 970
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		if args.watch:
			print("Loading Saved Network...")
			load_path = tf.train.latest_checkpoint(self.path)
			self.saver.restore(self.sess, load_path)
			print("Network Loaded")		
		else:
			self.sess.run(tf.initialize_all_variables())
			print("Network Initialized")
			self.summary_writer = tf.train.SummaryWriter('../records/' + args.game + '/' + args.agent_type + '/' + args.agent_name + '/params', self.sess.graph_def)


	def conv_relu(self, policy_input, target_input, kernel_shape, stride, layer_num):
		''' Build a convolutional layer

		Args:
			input_layer: input to convolutional layer - must be 4d
			target_input: input to layer of target network - must also be 4d
			kernel_shape: tuple for filter shape: (filter_height, filter_width, in_channels, out_channels)
			stride: tuple for stride: (1, vert_stride. horiz_stride, 1)
		'''
		name = 'conv' + str(layer_num + 1)
		with tf.variable_scope(name):

			# fan_in = tf.reduce_prod(tf.slice(policy_input.get_shape(), [1], [-1]))
			weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01), name=(name + "_weights"))
			# weights = self.get_weights(kernel_shape, fan_in, name + "_weights")
			biases = tf.Variable(tf.fill([kernel_shape[-1]], 0.1), name=(name + "_biases"))
			# biases = self.get_biases([kernel_shape[-1]], fan_in, name + "_biases")

			activation = tf.nn.relu(tf.nn.conv2d(policy_input, weights, stride, 'VALID') + biases)

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.nn.relu(tf.nn.conv2d(target_input, target_weights, stride, 'VALID') + target_biases)

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)
			self.param_names.append(name + "_weights")
			self.param_names.append(name + "_biases")

			return [activation, target_activation]


	def dense_relu(self, policy_input, target_input, shape, layer_num):
		''' Build a fully-connected relu layer 

		Args:
			input_layer: input to dense layer
			target_input: input to layer of target network
			shape: tuple for weight shape (num_input_nodes, num_layer_nodes)
		'''
		name = 'dense' + str(layer_num + 1)
		with tf.variable_scope(name):

			# fan_in = tf.reduce_prod(tf.slice(policy_input.get_shape(), [1], [-1]))
			weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
			# weights = self.get_weights(shape, fan_in, name + "_weights")
			biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))
			# biases = self.get_biases([shape[-1]], fan_in, name + "_biases")

			activation = tf.nn.relu(tf.matmul(policy_input, weights) + biases)

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.nn.relu(tf.matmul(target_input, target_weights) + target_biases)

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)
			self.param_names.append(name + "_weights")
			self.param_names.append(name + "_biases")

			return [activation, target_activation]


	def dense_linear(self, policy_input, target_input, shape):
		''' Build the fully-connected linear output layer 

		Args:
			input_layer: last hidden layer
			target_input: last hidden layer of target network
			shape: tuple for weight shape (num_input_nodes, num_actions)
		'''
		name = 'q_layer'
		with tf.variable_scope(name):

			# fan_in = tf.reduce_prod(tf.slice(policy_input.get_shape(), [1], [-1]))
			weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
			# weights = self.get_weights(shape, fan_in, name + "_weights")
			biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))
			# biases = self.get_biases([shape[-1]], fan_in, name + "_biases")


			activation = tf.matmul(policy_input, weights) + biases

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.matmul(target_input, target_weights) + target_biases

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)
			self.param_names.append(name + "_weights")
			self.param_names.append(name + "_biases")

			return [activation, target_activation]



	def inference(self, obs):
		''' Get state-action value predictions for an observation 

		Args:
			observation: the observation
		'''

		return np.squeeze(self.sess.run(self.policy_q_layer, feed_dict={self.observation:obs}))


	def build_loss(self, error_clip, num_actions, double_dqn):
		''' build loss graph '''
		with tf.name_scope("loss"):

			predictions = tf.reduce_sum(tf.mul(self.policy_q_layer, self.actions), 1)
			
			max_action_values = None
			if double_dqn: # Double Q-Learning:
				max_actions = tf.to_int32(tf.argmax(self.policy_q_layer, 1))
				# tf.gather doesn't support multidimensional indexing yet, so we flatten output activations for indexing
				indices = tf.range(0, tf.size(max_actions) * num_actions, num_actions) + max_actions
				max_action_values = tf.gather(tf.reshape(self.target_q_layer, shape=[-1]), indices)
			else:
				max_action_values = tf.reduce_max(self.target_q_layer, 1)

			targets = tf.stop_gradient(self.rewards + (self.discount_factor * max_action_values * self.terminals))

			difference = tf.abs(predictions - targets)

			if error_clip >= 0:
				quadratic_part = tf.clip_by_value(difference, 0.0, error_clip)
				linear_part = difference - quadratic_part
				errors = (0.5 * tf.square(quadratic_part)) + (error_clip * linear_part)
			else:
				errors = (0.5 * tf.square(difference))

			return tf.reduce_sum(errors)


	def train(self, o1, a, r, o2, t):
		''' train network on batch of experiences

		Args:
			o1: first observations
			a: actions taken
			r: rewards received
			o2: succeeding observations
		'''

		loss = self.sess.run([self.train_op, self.loss], 
			feed_dict={self.observation:o1, self.actions:a, self.rewards:r, self.next_observation:o2, self.terminals:t})[1]

		self.total_updates += 1
		if self.total_updates % self.target_update_frequency == 0:
			self.sess.run(self.update_target)

		return loss


	def save_model(self, epoch):

		self.saver.save(self.sess, self.path + '/' + self.name + '.ckpt', global_step=epoch)


	def build_rmsprop_optimizer(self, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip, version):

		with tf.name_scope('rmsprop'):
			optimizer = None
			if version == 'rmsprop':
				optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)
			elif version == 'graves_rmsprop':
				optimizer = tf.train.GradientDescentOptimizer(learning_rate)

			grads_and_vars = optimizer.compute_gradients(self.loss)
			grads = [gv[0] for gv in grads_and_vars]
			params = [gv[1] for gv in grads_and_vars]

			if gradient_clip > 0:
				grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

			if version == 'rmsprop':
				return optimizer.apply_gradients(zip(grads, params))
			elif version == 'graves_rmsprop':
				square_grads = [tf.square(grad) for grad in grads]

				avg_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]
				avg_square_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]

				update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])) 
					for grad_pair in zip(avg_grads, grads)]
				update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))) 
					for grad_pair in zip(avg_square_grads, grads)]
				avg_grad_updates = update_avg_grads + update_avg_square_grads

				rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
					for avg_grad_pair in zip(avg_grads, avg_square_grads)]


				rms_updates = [grad_rms_pair[0] / grad_rms_pair[1] for grad_rms_pair in zip(grads, rms)]
				train = optimizer.apply_gradients(zip(rms_updates, params))

				return tf.group(train, tf.group(*avg_grad_updates))


	def get_weights(self, shape, fan_in, name):
		std = 1 / tf.sqrt(tf.to_float(fan_in))
		return tf.Variable(tf.random_uniform(shape, minval=(0 - std), maxval=std), name=name)

	def get_biases(self, shape, fan_in, name):
		std = 1 / tf.sqrt(tf.to_float(fan_in))
		return tf.Variable(tf.fill(shape, std), name=name)

	def record_params(self, step):
		summary_string = self.sess.run(self.param_summaries)
		self.summary_writer.add_summary(summary_string, step)