import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO) 
from sample import VideoCaptionSample
from replay import Sample
import numpy as np 

class MIXER(object):

	def __init__(self, FLAGS, num_actions, start_id, end_id):
		
		self.num_actions = num_actions
		# self.target_model_frequency = FLAGS.target_model_frequency
		self.hidden_size = FLAGS.hidden_size
		self.dim = FLAGS.dim
		self.encoder_input_len = FLAGS.encoder_input_len
		self.encoder_input_size = FLAGS.encoder_input_size
		self.max_step = FLAGS.max_step
		self.start_id = start_id
		self.end_id = end_id

		self.learning_rate = FLAGS.learning_rate
		self.gamma = FLAGS.gamma

		self.log = FLAGS.log

		# tf.set_random_seed(1126)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True

		self.config = config

		# x is placeholder of video frames, 
		# y is output label sequence distribution,
		self.encoder_inputs, self.xent_dict, self.rein_dict = self.build_network('policy', True)
		trainable_variables = tf.global_variables()
		
		self.encoder_inputs_target, self.xent_dict_target, self.rein_dict_target = self.build_network('target', False)
		
		# we can assign a non-trainable LSTMCell QQ

		# build variable copy ops (policy to target)
		self.update_target = []
		all_variables = tf.global_variables()
		assert(len(all_variables) == 2*len(trainable_variables), "number of all_variables must be 2 * number of trainable_variables")
		
		for i, v in enumerate(all_variables):
			if v not in trainable_variables:
				self.update_target.append(v.assign(all_variables[i-len(trainable_variables)]))

		self.update_output_weights = []
		self.update_output_weights.append(self.rein_dict['W_rein'].assign(self.xent_dict['W_xent']))
		self.update_output_weights.append(self.rein_dict['b_rein'].assign(self.xent_dict['b_xent']))

		# loss function part
		# self.y_, self.r_, self.xent_loss, self.rein_loss, self.loss, self.bleu, self.train_step = 
		self.xent_dict, self.rein_dict = self.build_loss_network(self.xent_dict, self.rein_dict)

		self.bleu_score, self.bleu_preds, self.bleu_labels = self.bleu_network(self.end_id)

		self.f1_score, self.f1_preds, self.f1_labels = self.f1_network(self.end_id)

		self.saver = tf.train.Saver(max_to_keep=25)

		self.sv = tf.train.Supervisor(logdir=self.log, saver=self.saver)




	def build_network(self, name, trainable):


		# encoder part
		encoder_inputs, en_outputs, en_states = self.build_encoder_network(name, trainable)

		# decoder part
		xent_dict, rein_dict = self.build_decoder_network(name, trainable, en_outputs, en_states, self.start_id)

		return encoder_inputs, xent_dict, rein_dict



	def build_encoder_network(self, name, trainable):

		
		with tf.variable_scope("encoder_" + name):
			encoder_inputs = tf.placeholder(tf.float32, [None, self.encoder_input_len, self.encoder_input_size], name='encoder_inputs')
			cell = tf.contrib.rnn.BasicLSTMCell(
				self.hidden_size, forget_bias=0.0, state_is_tuple=True)

			# init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32) #batch_size
			# outputs: (None, len, size), state: (c(None, size), h(None, size))
			outputs, states = tf.nn.dynamic_rnn(cell, 
												encoder_inputs,  
												sequence_length=tf.fill([tf.shape(encoder_inputs)[0]], self.encoder_input_len), 
												dtype=tf.float32, 
												# initial_state=init_state, 
												scope='rnn_encoder')
			return encoder_inputs, outputs, states
	
	def build_decoder_network(self, name, trainable, en_outputs, en_states, start_id):
		
		with tf.variable_scope("decoder_" + name) as scope:

			# decoder input
			decoder_inputs = tf.placeholder(tf.int32, [None, self.max_step], name='decoder_inputs')

			# embedding = tf.Variable(tf.truncated_normal([self.num_actions, self.dim], stddev=0.1))
			embedding = tf.get_variable("embedding", [self.num_actions, self.dim], dtype=tf.float32)
			decoder_embed_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)

			# LSTM part
			# output_fn = lambda x:tf.contrib.layers.linear(x, self.num_actions, biases_initializer=tf.constant_initializer(0), trainable=True, scope=scope)
			W_xent = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_actions], stddev=0.1))
			b_xent = tf.Variable(tf.truncated_normal([self.num_actions], stddev=0.1))

			W_rein = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_actions], stddev=0.1))
			b_rein = tf.Variable(tf.truncated_normal([self.num_actions], stddev=0.1))

			# output_fn_rein = lambda x:tf.contrib.layers.linear(x, self.num_actions, biases_initializer=tf.constant_initializer(0), scope=scope)
			# output_fn_rein = lambda x: tf.contrib.layers.fully_connected(inputs=x, num_outputs=self.num_actions, activation_fn=None, trainable=True, biases_initializer=tf.constant_initializer(0), scope=scope)


			cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

			# attention
			attention_keys, attention_values, attention_score_fn, attention_construct_fn = tf.contrib.seq2seq.prepare_attention(
													attention_states=en_outputs,
													attention_option='bahdanau',
													num_units=self.hidden_size)

			
			# to implement MIXER, we need a placeholder to cut the border of XENT and REINFORCE predict
			# XENT part: we get inputs from decoder_inputs
			# REINFORCE part: we get inputs from last output.

			# because we have the same border in the same batch, so we only need a scalar
			xent_border = tf.placeholder(tf.int32)


			logits = []
			c_state, h_state = en_states
			cell_output = en_outputs[:, -1]
			rein_input_vector = tf.nn.embedding_lookup(embedding, tf.fill([tf.shape(en_outputs)[0]], start_id))

			border_c_state = c_state
			border_h_state = h_state

			for step in range(self.max_step):
				if step > 0: scope.reuse_variables()
				context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
				xent_input_vector = decoder_embed_inputs[:, step, :]
				word_vector = tf.cond(xent_border > step , lambda: xent_input_vector, lambda: rein_input_vector)
				input_vector = tf.concat([word_vector, context_vector], axis=-1)
				cell_output, (c_state, h_state) = cell(input_vector, tf.contrib.rnn.LSTMStateTuple(c_state, h_state))
				border_c_state, border_h_state = tf.cond(xent_border > step, lambda: (c_state, h_state), lambda: (border_c_state, border_h_state))

				logit = tf.cond(xent_border > step, lambda: tf.matmul(cell_output, W_xent) + b_xent, lambda: tf.matmul(cell_output, W_rein) + b_rein)

				logits.append(logit)

				# get next input word vector for rein
				max_word_idx = tf.argmax(logit, axis=1)
				rein_input_vector = tf.nn.embedding_lookup(embedding, max_word_idx)

			# logits: [(1, batch, vocab)] * max_step

			logits = tf.transpose(tf.stack(logits), [1, 0, 2])
			# now logits: (batch, max_step, vocab)


			# for greedy inference, we only give xent_border as 0

			y = tf.argmax(logits, axis=2)


			# build reinforce interface

			input_c_state = tf.placeholder(tf.float32, [None, self.hidden_size])
			input_h_state = tf.placeholder(tf.float32, [None, self.hidden_size])

			sample_or_max = tf.placeholder(tf.bool)

			scope.reuse_variables()
			logit = tf.matmul(input_h_state, W_rein) + b_rein

			# word_idx = tf.cond(sample_or_max, lambda: tf.multinomial(logit, 1)[:, 0], lambda: tf.argmax(logit, axis=1))

			values, indices = tf.nn.top_k(logit, 3)
			value_bound = tf.reduce_min(values, axis=1)
			prob = tf.cast(tf.greater_equal(logit, value_bound), tf.float32)

			word_idx = tf.cond(sample_or_max, lambda: tf.multinomial(prob, 1)[:, 0], lambda: tf.argmax(logit, axis=1))

			rein_input_vector = tf.nn.embedding_lookup(embedding, word_idx)
			context_vector = attention_construct_fn(input_h_state, attention_keys, attention_values)
			# freezed_context_vector = tf.stop_gradient(context_vector)
			input_vector = tf.concat([rein_input_vector, context_vector], axis=-1)
			cell_output, (c_state, h_state) = cell(input_vector, tf.contrib.rnn.LSTMStateTuple(input_c_state, input_h_state))
			scope.reuse_variables()
			
			logit = tf.matmul(cell_output, W_rein) + b_rein
			r = logit
			output_y = tf.argmax(logit, axis=1)

			xent_dict = {
				'decoder_inputs':decoder_inputs,
				'logits':logits,
				'border_c_state':border_c_state,
				'border_h_state':border_h_state,
				'y':y,
				'xent_border':xent_border,
				'W_xent':W_xent,
				'b_xent':b_xent
			}

			rein_dict = {
				'input_c_state':input_c_state,
				'input_h_state':input_h_state,
				'output_c_state':c_state,
				'output_h_state':h_state,
				'r':r,
				'y':output_y,
				'sample_or_max':sample_or_max,
				'W_rein':W_rein,
				'b_rein':b_rein
			}




			return xent_dict, rein_dict

	def build_loss_network(self, xent_dict, rein_dict):

		with tf.name_scope("loss"):


			# xent part

			# get ground truth sequence
			y_ = tf.placeholder(tf.int32, shape=[None, self.max_step], name='y_')


			xent_logits = xent_dict['logits'][:, :xent_dict['xent_border'], :]
			xent_labels = y_[:, :xent_dict['xent_border']]
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=xent_labels, logits=xent_logits)
			position_weight = tf.constant([self.max_step/np.log(i+2) for i in range(self.max_step)], dtype=tf.float32)
			xent_loss = tf.reduce_mean(cross_entropy * position_weight[:xent_dict['xent_border']])
			total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=xent_dict['logits']) * position_weight)

			# rein part

			a = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='a')
			r_ = tf.placeholder(tf.float32, shape=[None], name='r_')

			a_r_ = tf.multiply(r_[:, None], a)

			rein_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(a_r_, rein_dict['r']))

			# y_a = tf.reduce_sum(tf.multiply(rein_dict['r'], a), 1)

			# rein_loss = tf.reduce_mean(tf.square(y_a - r_))

			optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.95, epsilon=.01)


			tvars = tf.trainable_variables()

			xent_grads, _ = tf.clip_by_global_norm(tf.gradients(xent_loss, tvars), 5)

			xent_train_step = optimizer.apply_gradients(zip(xent_grads, tvars))

			# tvars = [rein_dict['W_rein'], rein_dict['b_rein']]

			rein_grads, _ = tf.clip_by_global_norm(tf.gradients(rein_loss, tvars), 5)

			rein_train_step = optimizer.apply_gradients(zip(rein_grads, tvars))
			# xent_train_step = optimizer.minimize(xent_loss)

			# rein_train_step = optimizer.minimize(rein_loss)


			# update dict
			xent_dict['y_'] = y_
			xent_dict['loss'] = xent_loss
			xent_dict['total_loss'] = total_loss
			xent_dict['train_step'] = xent_train_step

			rein_dict['a'] = a 
			rein_dict['r_'] = r_ 
			rein_dict['loss'] = rein_loss
			rein_dict['train_step'] = rein_train_step


			return xent_dict, rein_dict


	def bleu_network(self, end_id):

		# bleu-1: unigram bleu score

		y = tf.placeholder(tf.int32, [None, self.max_step])
		y_ = tf.placeholder(tf.int32, [None, self.max_step])

		# use y and y_ to compute BLEU@1

		# BLEU@1 = BP * Precision

		# BP = 1 if len(y) > len(y_) else exp(1 - len(y_) / len(y))

		# Precision = number of y in y_ / len(y)

		# pad end_id to deal with seq whose length == max_step
		def pad(x, end_id):
			return tf.concat([x, end_id * tf.ones([tf.shape(x)[0], 1], dtype=tf.int32)], axis=1)

		pad_y = pad(y, end_id)
		pad_y_ = pad(y_, end_id)	

		# get both valid length by end_id
		def find_end(pad, end_id):
			s_pad = tf.where(tf.equal(pad, end_id), tf.ones_like(pad, dtype=tf.float64), tf.zeros_like(pad, dtype=tf.float64))
			m_pad = tf.argmax(s_pad, axis=1)
			return m_pad

		end_y = find_end(pad_y, end_id)
		end_y_ = find_end(pad_y_, end_id)

		# compute BP
		# bp: [None]
		
		bp = tf.where(tf.less(end_y_, end_y), tf.ones_like(end_y, dtype=tf.float64), tf.exp(tf.constant(1.0, dtype=tf.float64) - end_y_ / end_y))

		

		# compute Precision

		def bow(x, len_x, max_step, num_actions, end_id):
			one_hot = tf.one_hot(x, num_actions)
			mask = tf.where(tf.sequence_mask(len_x, maxlen=max_step), x, end_id * tf.ones_like(x))
			bin = tf.reduce_sum(tf.one_hot(mask, num_actions), axis=1)
			bow = tf.concat([bin[:,:end_id], bin[:,(end_id+1):]], axis=1)
			return bow

		# def bow_bool(x, len_x, max_step, num_actions, end_id):
		# 	one_hot = tf.one_hot(x, num_actions)
		# 	mask = tf.where(tf.sequence_mask(len_x, maxlen=max_step), x, end_id * tf.ones_like(x))
		# 	bin = tf.reduce_sum(tf.one_hot(mask, num_actions), axis=1)
		# 	bow_bool = tf.cast(tf.concat([bin[:,:end_id], bin[:,(end_id+1):]], axis=1), tf.bool)
		# 	return bow_bool

		# max_step + because padding
		bow_y = bow(pad_y, end_y, self.max_step + 1, self.num_actions, end_id)

		bow_y_ = bow(pad_y_, end_y_, self.max_step + 1, self.num_actions, end_id)

		match_num = tf.reduce_sum(tf.cast(tf.where(tf.greater(bow_y_, 0), bow_y, tf.zeros_like(bow_y)), tf.float64), axis=1) 

		precision = match_num / tf.cast(end_y, tf.float64)

		bleu = tf.where(tf.less(tf.zeros_like(end_y), end_y), bp * precision, tf.zeros_like(end_y, dtype=tf.float64))

		return bleu, y, y_

	def f1_network(self, end_id):

		y = tf.placeholder(tf.int32, [None, self.max_step])
		y_ = tf.placeholder(tf.int32, [None, self.max_step])

		# use y and y_ to compute BLEU@1

		# BLEU@1 = BP * Precision

		# BP = 1 if len(y) > len(y_) else exp(1 - len(y_) / len(y))

		# Precision = number of y in y_ / len(y)

		# pad end_id to deal with seq whose length == max_step
		def pad(x, end_id):
			return tf.concat([x, end_id * tf.ones([tf.shape(x)[0], 1], dtype=tf.int32)], axis=1)

		pad_y = pad(y, end_id)
		pad_y_ = pad(y_, end_id)	

		# get both valid length by end_id
		def find_end(pad, end_id):
			s_pad = tf.where(tf.equal(pad, end_id), tf.ones_like(pad, dtype=tf.float64), tf.zeros_like(pad, dtype=tf.float64))
			m_pad = tf.argmax(s_pad, axis=1)
			return m_pad

		end_y = find_end(pad_y, end_id)
		end_y_ = find_end(pad_y_, end_id)

		# compute BP
		# bp: [None]
		
		bp = tf.where(tf.less(end_y_, end_y), tf.ones_like(end_y, dtype=tf.float64), tf.exp(tf.constant(1.0, dtype=tf.float64) - end_y_ / end_y))

		

		# compute Precision

		def bow(x, len_x, max_step, num_actions, end_id):
			one_hot = tf.one_hot(x, num_actions)
			mask = tf.where(tf.sequence_mask(len_x, maxlen=max_step), x, end_id * tf.ones_like(x))
			bin = tf.reduce_sum(tf.one_hot(mask, num_actions), axis=1)
			bow = tf.concat([bin[:,:end_id], bin[:,(end_id+1):]], axis=1)
			return bow

		# def bow_bool(x, len_x, max_step, num_actions, end_id):
		# 	one_hot = tf.one_hot(x, num_actions)
		# 	mask = tf.where(tf.sequence_mask(len_x, maxlen=max_step), x, end_id * tf.ones_like(x))
		# 	bin = tf.reduce_sum(tf.one_hot(mask, num_actions), axis=1)
		# 	bow_bool = tf.cast(tf.concat([bin[:,:end_id], bin[:,(end_id+1):]], axis=1), tf.bool)
		# 	return bow_bool

		# max_step + because padding
		bow_y = bow(pad_y, end_y, self.max_step + 1, self.num_actions, end_id)

		bow_y_ = bow(pad_y_, end_y_, self.max_step + 1, self.num_actions, end_id)

		# dochi? hit one == all, or hit one == one?
		# match_num = tf.reduce_sum(tf.cast(tf.where(tf.greater(bow_y_, 0), bow_y, tf.zeros_like(bow_y)), tf.float64), axis=1) 
		match_num = tf.reduce_sum(tf.cast(tf.minimum(bow_y_, bow_y), tf.float64), axis=1)

		precision = tf.where(tf.greater(end_y, 0), match_num / tf.cast(end_y, tf.float64), tf.zeros_like(match_num))

		recall = tf.where(tf.greater(end_y_, 0), match_num / tf.cast(end_y_, tf.float64), tf.zeros_like(match_num))


		p_r = precision + recall
		pr = precision * recall


		f1 = tf.where(tf.greater(p_r, 0), 2 * pr / p_r, tf.zeros_like(p_r))

		# bleu = tf.where(tf.less(tf.zeros_like(end_y), end_y), bp * precision, tf.zeros_like(end_y, dtype=tf.float64))

		return f1, y, y_

	def bleu(self, sess, preds, labels):

		bleu_score = sess.run(self.bleu_score, feed_dict={self.bleu_preds:preds, self.bleu_labels:labels})
		return bleu_score 

	def f1(self, sess, preds, labels):
		f1_score = sess.run(self.f1_score, feed_dict={self.f1_preds:preds, self.f1_labels:labels})
		return f1_score

	def inference(self, sess, encoder_inputs, input_c_states, input_h_states):

		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.rein_dict['input_c_state']:input_c_states,
			self.rein_dict['input_h_state']:input_h_states,
			self.rein_dict['sample_or_max']:True
		}

		ops = [self.rein_dict['output_c_state'], self.rein_dict['output_h_state'], self.rein_dict['y']]
		c_states, h_states, y =  sess.run(ops, feed_dict=feed_dict)

		return c_states, h_states, y

	def eval(self, sess, batch, xent_border):
		# batch = batch of VideoCaptionSample

		encoder_inputs = [b.frames for b in batch]
		decoder_inputs = [[self.start_id] * self.max_step for b in batch]
		y_ = [b.labels for b in batch]
		

		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.xent_dict['decoder_inputs']:decoder_inputs,
			self.xent_dict['y_']:y_,
			self.xent_dict['xent_border']:0
		}
		ops = self.xent_dict['y']
		y = sess.run(ops, feed_dict=feed_dict)

		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.xent_dict['decoder_inputs']:decoder_inputs,
			self.xent_dict['y_']:y_,
			self.xent_dict['xent_border']:xent_border
		}
		ops = self.xent_dict['loss']
		loss = sess.run(ops, feed_dict=feed_dict)

		
		return loss, y


	def predict(self, sess, batch):

		encoder_inputs = [b.frames for b in batch]
		decoder_inputs = [[self.start_id] * self.max_step for b in batch]
		y_ = [b.labels for b in batch]

		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.xent_dict['decoder_inputs']:decoder_inputs,
			self.xent_dict['y_']:y_,
			self.xent_dict['xent_border']:0
		}

		y = sess.run(self.xent_dict['y'], feed_dict=feed_dict)
		return y 

	def train_xent(self, sess, batch, xent_border, train=True):

		# batch = batch of VideoCaptionSample

		encoder_inputs = [b.frames for b in batch]
		decoder_inputs = [[self.start_id] + b.labels[:-1] for b in batch]
		y_ = [b.labels for b in batch]
		

		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.xent_dict['decoder_inputs']:decoder_inputs,
			self.xent_dict['y_']:y_,
			self.xent_dict['xent_border']:xent_border
		}

		if train:
			ops = [self.xent_dict['train_step'], self.xent_dict['loss'], self.xent_dict['border_c_state'], self.xent_dict['border_h_state'], self.xent_dict['y']]
			_, loss, c_state, h_state, y = sess.run(ops, feed_dict=feed_dict)
		else:
			ops = [self.xent_dict['loss'], self.xent_dict['border_c_state'], self.xent_dict['border_h_state'], self.xent_dict['y']]
			loss, c_state, h_state, y = sess.run(ops, feed_dict=feed_dict)

		# end after xent_border 
		# y[:, xent_border-1] = self.end_id
		return loss, c_state, h_state, y

	def target_model_update(self, sess):
		sess.run(self.update_target)

	def initial_rein_weight_update(self, sess):
		sess.run(self.update_output_weights)

	def train_rein(self, sess, batch):

		encoder_inputs = [b.frames for b in batch]
		# target network
		# the next step
		target_c_states = [b.output_c_state for b in batch]
		target_h_states = [b.output_h_state for b in batch]
		target_r = sess.run(self.rein_dict_target['r'], 
			feed_dict={
			self.encoder_inputs_target:encoder_inputs, # for attention
			self.rein_dict_target['input_c_state']:target_c_states,
			self.rein_dict_target['input_h_state']:target_h_states,
			self.rein_dict_target['sample_or_max']:False}
			)

		# batch = batch of Replay
		
		c_states = [b.input_c_state for b in batch]
		h_states = [b.input_h_state for b in batch]
		a = np.zeros((len(batch), self.num_actions))
		r_ = np.zeros(len(batch))


		for i in range(len(batch)):
			a[i, batch[i].action] = 1
			if batch[i].terminal:
				r_[i] = batch[i].reward
			else:
				r_[i] = batch[i].reward + self.gamma * np.max(target_r[i])


		feed_dict = {
			self.encoder_inputs:encoder_inputs,
			self.rein_dict['input_c_state']:c_states,
			self.rein_dict['input_h_state']:h_states,
			self.rein_dict['a']:a,
			self.rein_dict['r_']:r_,
			self.rein_dict['sample_or_max']:False
		}
		ops = [self.rein_dict['train_step'], self.rein_dict['loss']]
		_, loss = sess.run(ops, feed_dict=feed_dict)

		return loss







