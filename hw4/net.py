import tensorflow as tf 
import numpy as np 
from tensorflow.python.framework import ops


def getSeluParameters(fixedpointMean=0,fixedpointVar=1):
    """ Finding the parameters of the SELU activation function. The function returns alpha and lambda for the desired fixed point. """
    
    import sympy
    from sympy import Symbol, solve, nsolve

    aa = Symbol('aa')
    ll = Symbol('ll')
    nu = fixedpointMean 
    tau = fixedpointVar 

    mean =  0.5*ll*(nu + np.exp(-nu**2/(2*tau))*np.sqrt(2/np.pi)*np.sqrt(tau) + \
                        nu*erf(nu/(np.sqrt(2*tau))) - aa*erfc(nu/(np.sqrt(2*tau))) + \
                        np.exp(nu+tau/2)*aa*erfc((nu+tau)/(np.sqrt(2*tau))))

    var = 0.5*ll**2*(np.exp(-nu**2/(2*tau))*np.sqrt(2/np.pi*tau)*nu + (nu**2+tau)* \
                          (1+erf(nu/(np.sqrt(2*tau)))) + aa**2 *erfc(nu/(np.sqrt(2*tau))) \
                          - aa**2 * 2 *np.exp(nu+tau/2)*erfc((nu+tau)/(np.sqrt(2*tau)))+ \
                          aa**2*np.exp(2*(nu+tau))*erfc((nu+2*tau)/(np.sqrt(2*tau))) ) - mean**2

    eq1 = mean - nu
    eq2 = var - tau

    res = nsolve( (eq2, eq1), (aa,ll), (1.67,1.05))
    return float(res[0]),float(res[1])


class RNNEncoder(object):

	def __init__(self, name, num_step, hidden_size):
		self.name = '{}/rnn_encoder'.format(name)
		self.hidden_size = hidden_size
		self.num_step = num_step

		# parameter for selu
		self.alpha, self.scale = 1.6732632423543774, 1.0507009873554802

	def selu(self, x):
		with ops.name_scope('elu') as scope:
		    alpha = self.alpha
		    scale = self.scale
		    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

	def __call__(self, encoder_inputs, embedding, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()

			cell = tf.contrib.rnn.BasicLSTMCell(
				self.hidden_size, forget_bias=1.0, state_is_tuple=True, 
				activation=self.selu 
				)

			encoder_embed_inputs = tf.nn.embedding_lookup(embedding, encoder_inputs)

			# init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32) #batch_size

			# outputs: (None, len, size), state: (c(None, size), h(None, size))
			
			outputs, state = tf.nn.dynamic_rnn(cell, 
												encoder_embed_inputs,  
												sequence_length=tf.fill([tf.shape(encoder_embed_inputs)[0]], self.num_step), 
												dtype=tf.float32, 
												# initial_state=init_state
												)

			return state

	@property 
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class RNNDecoder(object):

	def __init__(self, name, num_step, hidden_size, vocab_dim, output_size, start_id, end_id, mode='reinforce'):
		self.name = '{}/rnn_decoder'.format(name)
		self.hidden_size = hidden_size
		self.num_step = num_step
		self.vocab_dim = vocab_dim
		self.output_size = output_size
		self.start_id = start_id 
		self.end_id = end_id
		self.mode = mode

		# parameter for selu
		self.alpha, self.scale = 1.6732632423543774, 1.0507009873554802
		
	def selu(self, x):
		with ops.name_scope('elu') as scope:
		    alpha = self.alpha
		    scale = self.scale
		    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


	def __call__(self, encoder_state,  greedy_controller, decoder_inputs=None, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()

			embedding = tf.get_variable("decoder_embedding", [self.output_size, self.vocab_dim], dtype=tf.float32)
			self.embedding = embedding
			cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True,
				activation=self.selu 
				)

			# output_fn = lambda x:tf.layers.dense(x, self.output_size, bias_initializer=tf.constant_initializer(0))
			
			W = tf.get_variable("output_weights", [self.hidden_size, self.output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("output_biases", [self.output_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
			# def output_fn(x):
			# 	return tf.layers.dense(
			# 		x, 
			# 		self.output_size,
			# 		bias_initializer=tf.constant_initializer(0)
			# 	)

			if self.mode == 'actor-critic':
				W_c = tf.get_variable("critic_weights", [self.hidden_size, self.output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				b_c = tf.get_variable("critic_biases", [self.output_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
				# W_c2 = tf.get_variable("critic_weights2", [1024, self.output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				# b_c2 = tf.get_variable("critic_biases2", [self.output_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
				# W_c3 = tf.get_variable("critic_weights3", [1024, 1024], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				# b_c3 = tf.get_variable("critic_biases3", [1024], dtype=tf.float32, initializer=tf.constant_initializer(0))
				# W_c4 = tf.get_variable("critic_weights4", [1024, self.output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				# b_c4 = tf.get_variable("critic_biases4", [self.output_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
				critic_outputs = []

			decoder_outputs = []


			

			if decoder_inputs is not None:


				decoder_embed_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)
				state = encoder_state
						
				for step in range(self.num_step):
					input_vector = decoder_embed_inputs[:, step, :]

					if step > 0: vs.reuse_variables()
					decoder_output, state = cell(input_vector, state, scope=vs)
					
					decoder_output2 = tf.matmul(decoder_output, W) + b 

					if self.mode == 'actor-critic':

						# decoder_output = tf.concat([decoder_output, self.selu(decoder_output2)], axis=1)
						critic_output = tf.matmul(decoder_output, W_c) + b_c
						# critic_output = tf.matmul(critic_output, W_c2) + b_c2
					
						critic_outputs.append(critic_output)

					decoder_outputs.append(decoder_output2)
					
				decoder_outputs = tf.transpose(tf.stack(decoder_outputs), [1, 0, 2])
				input_words = decoder_inputs
				output_words = tf.argmax(decoder_outputs, axis=2)

				if self.mode == 'actor-critic':
					critic_outputs = tf.transpose(tf.stack(critic_outputs), [1, 0, 2])
					return decoder_outputs, input_words, output_words, critic_outputs
			else:

				
				
				output_words = []
				input_vector = tf.nn.embedding_lookup(embedding, tf.fill([tf.shape(encoder_state.h)[0]], self.start_id))
				
				state = encoder_state
		
				for step in range(self.num_step):
					if step > 0: 
						vs.reuse_variables()
						
					# print(state)
					decoder_output, state = cell(input_vector, state, scope=vs)
					# print(state)
					
					decoder_output = tf.matmul(decoder_output, W) + b 
		
					decoder_outputs.append(decoder_output)
					# multinomail output
					# if step == 0:
					word_index = tf.cond(greedy_controller, lambda: tf.argmax(decoder_output, axis=1), lambda: tf.multinomial(decoder_output, 1)[:, 0])
					# word_index = tf.cond(greedy_controller, lambda: tf.argmax(decoder_output, axis=1), lambda: tf.multinomial(tf.ones_like(decoder_output), 1)[:, 0])
					
					# else:
					# word_index = tf.argmax(decoder_output, axis=1)

					output_words.append(word_index)
					input_vector = tf.nn.embedding_lookup(embedding, word_index)
					

				decoder_outputs = tf.transpose(tf.stack(decoder_outputs), [1, 0, 2])
				output_words = tf.transpose(tf.stack(output_words))

				return output_words
		

		

	@property 
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]



class Seq2seq(object):
	def __init__(self, name, num_step, hidden_size, vocab_size, vocab_dim, output_size, start_id, end_id, mode='reinforce'):
		self.name = '{}/seq2seq'.format(name)
		self.hidden_size = hidden_size
		self.num_step = num_step
		self.vocab_size = vocab_size
		self.vocab_dim = vocab_dim
		self.start_id = start_id 
		self.end_id = end_id
		self.output_size = output_size
		self.mode = mode

		self.rnn_encoder = RNNEncoder(self.name, self.num_step, self.hidden_size)
		self.rnn_decoder = RNNDecoder(self.name, self.num_step, self.hidden_size, self.vocab_dim, self.output_size, self.start_id, self.end_id, mode=mode)

	def __call__(self, encoder_inputs, greedy_controller, decoder_inputs=None, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()

			embedding = tf.get_variable("embedding", [self.vocab_size, self.vocab_dim], dtype=tf.float32)

		encoder_state = self.rnn_encoder(encoder_inputs, embedding, reuse=reuse)
		self.encoder_state = encoder_state

		if self.mode == 'reinforce':
			if decoder_inputs is not None:
				decoder_outputs, input_words, output_words = self.rnn_decoder(encoder_state, greedy_controller, decoder_inputs, reuse=reuse)
				self.decoder_outputs = decoder_outputs
			else:
				decoder_outputs, input_words, output_words = self.rnn_decoder(encoder_state, greedy_controller, reuse=reuse)
				self.decoder_outputs = decoder_outputs
			return decoder_outputs, input_words, output_words
		elif self.mode == 'actor-critic':
			if decoder_inputs is not None:
				decoder_outputs, input_words, output_words, critic_outputs = self.rnn_decoder(encoder_state, greedy_controller, decoder_inputs, reuse=reuse)
				self.decoder_outputs = decoder_outputs
				return decoder_outputs, input_words, output_words, critic_outputs
			else:
				output_words = self.rnn_decoder(encoder_state,  greedy_controller, reuse=reuse)
				
				return output_words

	@property 
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

	@property 
	def vars_actor(self):
		return [var for var in tf.global_variables() if self.name in var.name and 'critic' not in var.name]

	@property 
	def vars_critic(self):
		return [var for var in tf.global_variables() if self.name in var.name and 'critic' in var.name]


if __name__ == "__main__":

	import numpy as np 
	# re = RNNEncoder("test", 10, 20)
	# rd = RNNDecoder('test', 10, 20, 30, 0, 1)

	r = Seq2seq("test", 10, 20, 30, 4, 30, 0, 1, mode='actor-critic')



	encoder_inputs = tf.constant(np.arange(3*10).reshape(3, 10))
	# embedding = tf.constant(np.arange(30*4).reshape(30, 4).astype('float32'))
	decoder_inputs = tf.constant(np.arange(3*10).reshape(3, 10))


	decoder_outputs, input_words, output_words = r(encoder_inputs, decoder_inputs, reuse=False)
	print(decoder_outputs)
	print(input_words)
	print(output_words)
	print([var.name for var in r.vars])

	decoder_outputs, input_words, output_words = r(encoder_inputs)
	print(decoder_outputs)
	print(input_words)
	print(output_words)
	print([var.name for var in r.vars])

	# encoder_state = re(encoder_inputs, embedding, reuse=False)

	# decoder_outputs = rd(encoder_state, embedding, decoder_inputs, reuse=False)

	# print(encoder_state)
	# print(decoder_outputs)
	# print([var.name for var in rd.vars])

	# decoder_outputs = rd(encoder_state, embedding)
	# print(decoder_outputs)
	# print([var.name for var in rd.vars])


