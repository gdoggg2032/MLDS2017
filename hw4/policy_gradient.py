import tensorflow as tf
import numpy as np
import random

from net import Seq2seq
# from replay_memory import ReplayMemory 

import progressbar as pb

class PolicyGradientBase(object):

	# define basic function here

	def inference(self, sess, states, model='policy'):
		if model == 'target':
			opt = self.target_output_words
			greedy_controller=False
		else:
			opt = self.output_words
			greedy_controller = True

		output_words = sess.run(opt, feed_dict={
				self.states:states,
				self.greedy_controller:greedy_controller
			})
		return output_words

	def seq2seq_train(self, sess, data, length=None, train=True):

		states = [d.state for d in data]
		targets = [d.target for d in data]
		prev_targets = [d.prev_target for d in data]


		if train:

			_, loss = sess.run([self.opt_pretrain_actor, self.pretrain_actor_loss] , feed_dict={
					self.states:states,
					self.targets: targets,
					self.prev_targets: prev_targets,
					self.greedy_controller: True
				})
		else:
			loss = sess.run(self.pretrain_actor_loss , feed_dict={
					self.states:states,
					self.targets: targets,
					self.prev_targets: prev_targets,
					self.greedy_controller: True
				})
		# print(loss, length)
		return loss

	def seq2seq_loss(self, sess, data, actions):


		states = [d.state for d in data]
		targets = [d.target for d in data]
		prev_targets = [d.prev_target for d in data]

		loss = sess.run(self.pretrain_actor_loss_split, feed_dict={
					self.states:actions,
					self.targets: targets,
					self.prev_targets: prev_targets,
					self.greedy_controller: True
				})

		return loss

	def sentence_sim(self, sess, data, actions):

		targets = [d.target for d in data]

		sim = sess.run(self.sentence_similarity, feed_dict={
					self.targets: targets,
					self.taken_actions: actions
			})
		return sim
		


	def update_model(self, sess, data, actions, rewards, both=True):

		states = [d.state for d in data]
		targets = [d.target for d in data]
		prev_targets = [d.prev_target for d in data]
		prev_actions = [[self.start_id] + list(a[:-1]) for a in actions]



		# _, c_loss = sess.run([self.opt_critic, self.critic_loss], 
		# 	feed_dict={
		# 		self.states: states,
		# 		self.taken_actions: actions,
		# 		self.decoder_inputs: prev_actions,
		# 		self.discounted_rewards: rewards,
		# 		self.greedy_controller: True
		# 	})
		c_loss = 0


		if both:
			_, s = sess.run([self.opt_actor, self.joint_scores],
				feed_dict={
					self.states: states,
					self.targets: targets,
					self.prev_targets: prev_targets,
					self.taken_actions: actions,
					self.decoder_inputs: prev_actions,
					self.discounted_rewards: rewards,
					self.greedy_controller: True
				})
			return c_loss, s
		else:
			return c_loss





	

	def store_rollout(self, state, action, reward, target, done):
		self.replay_memory.add(state, action, reward, target, done)




class PolicyGradientReinforce(PolicyGradientBase):
	
	def __init__(self,
				 num_step,
				 hidden_size,
				 vocab_size,
				 vocab_dim,
				 start_id,
				 learning_rate,
				 clip_norm=5,
				 discount_factor=0.99,
				 replay_memory_size=1000):

		self.name = "pg_reinforce"

		# input: (None, num_step)
		# output: (None, num_step)

		# model components
		
		self.policy_net = Seq2seq("{}/policy".format(self.name), num_step, hidden_size, vocab_size, vocab_dim, vocab_size, start_id, mode='reinforce')

		self.replay_memory = ReplayMemory(memory_size=replay_memory_size, discount_factor=discount_factor)

		# training parameters
		self.num_step = num_step
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.vocab_dim = vocab_dim
		self.start_id = start_id
		self.learning_rate = learning_rate
		self.clip_norm = clip_norm

		# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		# place holders
		self.states = tf.placeholder(tf.int32, [None, num_step])
		self.taken_actions = tf.placeholder(tf.int32, [None, num_step])
		# notice that rewards is step-level for future extension
		self.discounted_rewards = tf.placeholder(tf.float32, [None, num_step])

		self.targets = tf.placeholder(tf.int32, [None, num_step])

		self.greedy_controller = tf.placeholder(tf.bool, [])

		self.decoder_inputs = tf.placeholder(tf.int32, [None, num_step])

		self.action_scores, _, self.output_words = self.policy_net(self.states, self.greedy_controller, reuse=False)

		# compute loss and gradients

		logprobs, _ ,_ = self.policy_net(self.states, self.greedy_controller, self.decoder_inputs)

		# logprobs: (None, num_step, vocab_size)

		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.taken_actions, logits=logprobs)
		self.pg_loss = tf.reduce_mean(tf.reduce_sum(self.discounted_rewards * cross_entropy, axis=1))


		self.loss = self.pg_loss #- 0.01 * cross_entropy

		gradients = optimizer.compute_gradients(self.loss)

		# clip gradients
		for i, (grad, var) in enumerate(gradients):
			if grad is not None:
				gradients[i] = (tf.clip_by_norm(grad, self.clip_norm), var)


		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.opt = optimizer.apply_gradients(gradients)

		# seq2seq part
		
		logits, _, _ = self.policy_net(self.states, self.greedy_controller, self.decoder_inputs)

		self.seq_weights = tf.placeholder(tf.float32, [None, num_step])

		self.seq2seq_loss = tf.contrib.seq2seq.sequence_loss(
			logits=logits, targets=self.targets,
			weights=self.seq_weights)

		gradients = optimizer.compute_gradients(self.seq2seq_loss)

		# clip gradients
		for i, (grad, var) in enumerate(gradients):
			if grad is not None:
				gradients[i] = (tf.clip_by_norm(grad, self.clip_norm), var)


		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.opt_seq2seq = optimizer.apply_gradients(gradients)


	@property 
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]


class PolicyGradientActorCritic(PolicyGradientBase):
	def __init__(self,
				 num_step,
				 hidden_size,
				 vocab_size,
				 vocab_dim,
				 start_id,
				 end_id,
				 learning_rate,
				 clip_norm=5,
				 discount_factor=0.99,
				 replay_memory_size=1000):

		self.name = "pg_ActorCritic"

		# input: (None, num_step)
		# output: (None, num_step)

		# model components

		self.actor_critic_net = Seq2seq("{}/policy".format(self.name), num_step, hidden_size, vocab_size, vocab_dim, vocab_size, start_id, end_id, mode='actor-critic')
		self.target_net = Seq2seq("{}/target".format(self.name), num_step, hidden_size, vocab_size, vocab_dim, vocab_size, start_id, end_id, mode='actor-critic')
		# self.replay_memory = ReplayMemory(memory_size=replay_memory_size, discount_factor=discount_factor)


		# training parameters
		self.num_step = num_step
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.vocab_dim = vocab_dim
		self.start_id = start_id
		self.learning_rate = learning_rate
		self.clip_norm = clip_norm

		optimizer_seq2seq = tf.train.AdamOptimizer(learning_rate=1e-3)
		optimizer_critic = tf.train.AdamOptimizer(learning_rate=1e-3)
		optimizer_actor = tf.train.AdamOptimizer(learning_rate=1e-4)

		# place holders
		self.states = tf.placeholder(tf.int32, [None, num_step])
		self.taken_actions = tf.placeholder(tf.int32, [None, num_step])
		self.targets = tf.placeholder(tf.int32, [None, num_step])
		self.prev_targets = tf.placeholder(tf.int32, [None, num_step])

		self.greedy_controller = tf.placeholder(tf.bool, [])

		# notice that rewards is step-level for future extension
		self.discounted_rewards = tf.placeholder(tf.float32, [None, num_step])

		self.decoder_inputs = tf.placeholder(tf.int32, [None, num_step])


		# inference net
		self.output_words = self.actor_critic_net(self.states, self.greedy_controller, reuse=False)
		

		# target inference net
		self.target_output_words = self.target_net(self.states, self.greedy_controller, reuse=False)

		# target computation net
		target_action_scores, _, _, target_critic_outputs = self.target_net(self.states, self.greedy_controller, self.decoder_inputs)
		# critic_outputs: (None, num_step, vocab_size)

		# target computation net
		action_scores, _, _, critic_outputs = self.actor_critic_net(self.states, self.greedy_controller, self.decoder_inputs)
		action_scores_target, _, _, _ = self.actor_critic_net(self.states, self.greedy_controller, self.prev_targets)
		# critic_outputs: (None, num_step, vocab_size)


		# Compute targets for the critic
		probs = tf.nn.softmax(action_scores)
		# probs: (None, num_step, vocab_size)
		target_probs = tf.nn.softmax(target_action_scores)



		# estimated_values = tf.reduce_sum(probs * critic_outputs, axis=2)
		estimated_values = tf.reduce_sum(critic_outputs * tf.one_hot(self.taken_actions, self.vocab_size), axis=2)
		self.estimated_values = estimated_values
		# estimated_values: (None, num_step)
		expected_values = tf.reduce_sum(target_probs * target_critic_outputs, axis=2)

		target_estimated_values = tf.reduce_sum(target_probs * target_critic_outputs, axis=2) 
		target_acc_estimated_values = tf.concat([target_estimated_values[:, 1:], tf.zeros_like(target_estimated_values[:, -1:])], axis=1) + self.discounted_rewards


		self.target_estimated_values = tf.reduce_sum(target_critic_outputs * tf.one_hot(self.taken_actions, self.vocab_size), axis=2)
		

		acc_discounted_rewards = []
		for i in range(self.num_step):
			r = tf.reduce_sum(self.discounted_rewards[:, i:], axis=1)[:, None]
			acc_discounted_rewards.append(r)
		acc_discounted_rewards = tf.concat(acc_discounted_rewards, axis=1)
	
		# should be (None, num_step)

		# Update the critic weights using the gradient

		lambda_c = 1e-3

		c_loss = tf.reduce_sum((critic_outputs - tf.reduce_mean(critic_outputs, axis=2)[:, :, None]) ** 2, axis=2)

		square_loss = tf.reduce_sum((estimated_values - target_acc_estimated_values) ** 2 + lambda_c * c_loss, axis=1)

		# regularization_critic = tf.reduce_mean(critic_outputs ** 2)
		lambda_r = 1e1

		self.critic_loss = tf.reduce_mean(square_loss) #+ lambda_r * regularization_critic

		lambda_ll = 1e-1

		# advantages = acc_discounted_rewards - estimated_values 
		advantages = target_acc_estimated_values - estimated_values



		pg_loss = tf.contrib.seq2seq.sequence_loss(
			logits=action_scores, targets=self.taken_actions,
			weights=self.discounted_rewards,
			average_across_timesteps=False
			)

		target_seq_scores = tf.reduce_sum(tf.nn.softmax(action_scores_target) * tf.one_hot(self.targets, self.vocab_size), axis=[1,2])
		# target_seq_scores = tf.reduce_sum(tf.nn.softmax(action_scores_target) * tf.one_hot(self.targets, self.vocab_size), axis=[1,2])
		# joint_scores = tf.reduce_sum(probs * critic_outputs, axis=[1, 2])
		# self.joint_scores = tf.reduce_mean(joint_scores)
		# cross_entropy = joint_scores + lambda_ll * target_seq_scores

		cross_entropy = tf.contrib.seq2seq.sequence_loss(
			logits=action_scores_target, targets=self.targets,
			weights=tf.ones_like(advantages),
		)

		# gradient ascent -> descent "-"
		# self.actor_loss = tf.reduce_mean(-cross_entropy)
		self.actor_loss = tf.reduce_mean(pg_loss) #- lambda_ll * cross_entropy
		self.joint_scores = self.actor_loss


		target_embed = tf.nn.embedding_lookup(self.actor_critic_net.rnn_decoder.embedding, self.targets)
		action_embed = tf.nn.embedding_lookup(self.actor_critic_net.rnn_decoder.embedding, self.taken_actions)


		

		# cos_sim = tf.losses.cosine_distance(target_embed, action_embed, dim=2)
		cos_sim = tf.reduce_sum(target_embed * action_embed, axis=2) / (tf.sqrt(tf.reduce_sum(target_embed**2, axis=2)) * tf.sqrt(tf.reduce_sum(action_embed**2, axis=2)))


		self.sentence_similarity = cos_sim


		# pretrain actor

		self.pretrain_actor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=action_scores_target))
		self.pretrain_actor_loss_split = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=action_scores_target)

		pretrain_actor_gradients = optimizer_seq2seq.compute_gradients(self.pretrain_actor_loss, self.actor_critic_net.vars_actor)

		critic_gradients = optimizer_critic.compute_gradients(self.critic_loss, self.actor_critic_net.vars_critic)
		actor_gradients = optimizer_actor.compute_gradients(self.actor_loss, self.actor_critic_net.vars_actor)

		# clip gradients
		
		for i, (grad, var) in enumerate(pretrain_actor_gradients):
			if grad is not None:
				pretrain_actor_gradients[i] = (tf.clip_by_norm(grad, self.clip_norm), var)

		# print("critic_gradients")
		# for i, (grad, var) in enumerate(critic_gradients):
		# 	if grad is not None:
		# 		print(grad, var.name)
		# 		critic_gradients[i] = (tf.clip_by_norm(grad, self.clip_norm), var)

		# print("actor_gradients")
		# for i, (grad, var) in enumerate(actor_gradients):
		# 	if grad is not None:
		# 		print(grad, var.name)
		# 		actor_gradients[i] = (tf.clip_by_norm(grad, self.clip_norm), var)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

			

			self.opt_pretrain_actor = optimizer_seq2seq.apply_gradients(pretrain_actor_gradients)

			self.opt_critic = optimizer_critic.apply_gradients(critic_gradients)

			self.opt_actor = optimizer_actor.apply_gradients(actor_gradients)


		self.gamma = tf.placeholder(tf.float32)
		self.update_target = [var_target.assign(var * self.gamma + var_target * (1 - self.gamma)) for var_target, var in zip(self.target_net.vars, self.actor_critic_net.vars)]


	def target_model_update(self, sess, rate):
		sess.run(self.update_target, feed_dict={self.gamma:rate})

	def get_predicted_taken_action_reward(self, sess, data, actions, model='policy'):

		states = [d.state for d in data]
		prev_actions = [[self.start_id] + list(a[:-1]) for a in actions]

		if model == 'policy':
			opt = self.estimated_values
		elif model == 'target':
			opt = self.target_estimated_values

		estimated_values = sess.run(opt, feed_dict={
				self.states: states,
				self.decoder_inputs: prev_actions,
				self.taken_actions: actions,
				self.greedy_controller: True
			})
		return estimated_values
	

	@property 
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]
















