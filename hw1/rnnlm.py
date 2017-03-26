import tensorflow as tf 
import sys
import argparse
import time
import numpy as np 
import progressbar as pb
import logging
logging.basicConfig(level=logging.INFO)
from array import array
import csv

from vocab import Vocab
import pickle

import nltk
import pandas as pd 



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument("--name", default="rnnlm_vanilla", type=str)
	parser.add_argument('--tr_data', default="./tr_data.pickle", type=str)
	parser.add_argument('--val_data', default="./val_data.pickle", type=str)
	# parser.add_argument('--tr_labels', default="./train-labels-idx1-ubyte", type=str)
	parser.add_argument('--te_data', default="./te_data.pickle", type=str)
	parser.add_argument('--vocab', default="./vocab", type=str)

	parser.add_argument("--batch_size", default=100, type=int)
	parser.add_argument("--seq_size", default=40, type=int)
	parser.add_argument("--epochs", default=5, type=int)
	parser.add_argument("--lr", default=1.0, type=float)
	parser.add_argument("--lr_decay", default=1 / 1.15, type=float)
	parser.add_argument("--max_grad_norm", default=5.0, type=float)
	parser.add_argument("--num_sampled", default=100, type=int)
	parser.add_argument("--dim", default=300, type=int)
	parser.add_argument("--delay", default=1, type=int)
	parser.add_argument("--num_layers", default=1, type=int)
	parser.add_argument("--hidden_size", default=256, type=int)

	parser.add_argument('--min_count', default=10, type=int)
	parser.add_argument('--embedding', default=None, type=str)


	parser.add_argument("--model", default="./model", type=str)
	parser.add_argument("--predict", default="./predict.csv", type=str)
	parser.add_argument("--mode", default=0, type=int)
	parser.add_argument("--log", default="./log", type=str)


	args = parser.parse_args()

	return args 


def load_bin_vec(fname, vocab, dim):
	"""
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	"""
	# word_vecs = {}
	s_time = time.time()
	embedding = np.random.uniform(-0.25,0.25,(vocab.vocab_size, dim))
	stemmer = nltk.porter.PorterStemmer()
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size


		print(binary_len, vocab_size, layer1_size)
		
		for line in range(vocab_size):
			word = []

			while True:
				ch = f.read(1)
				
				if ch == b' ':
					word = b''.join(word)
					word = word.decode("utf-8")
					break
				if ch != '\n':
					word.append(ch) 
							
			word = stemmer.stem(word) 
			if word in vocab.w2i:
				vec = np.fromstring(f.read(binary_len), dtype='float32')  
				# word_vecs[word] = vec
				embedding[vocab.w2i[word]] = vec
			else:
				f.read(binary_len)
	logging.info("loaded pre-train embedding time: {}".format(time.time() - s_time))
	return embedding

class RNN(object):
	def __init__(self, args):
		self.args = args
		self.load_data()
		self.index = 0
		self.index_valid = 0
		self.index_test = 0

		

	def load_data(self):

		logging.info("loading vocabulary")

		self.vocab = Vocab(self.args.vocab)
		

		logging.info("loading data")

		if self.args.mode % 2 == 0:
			# load training data
			self.train_data = pickle.load(open(self.args.tr_data, "rb"))
			# raw_text_list = open(self.args.tr_data, "r", errors="ignore").read().strip().split()
			
			# total_data_len = len(raw_text_list)
			# val_len = total_data_len // 10

			# raw_train_text_list = raw_text_list[val_len:]

			# self.vocab = Vocab(self.args.vocab, raw_train_text_list)
			# raw_data = np.array(self.vocab.encode_all(raw_text_list))
			
			# total_data_len = raw_data.shape[0]
			# val_len = total_data_len // 10

			# self.train_data = raw_data[val_len:]
			# print(self.train_data.shape)
			# data_len = self.train_data.shape[0]
			# batch_size = self.args.batch_size
			# batch_len = batch_size * self.args.seq_size
			# total_batch = data_len - batch_len + 1
			self.tr_total_batch = int(np.ceil(len(self.train_data) / self.args.batch_size))




			# # do validation
			self.val_data = pickle.load(open(self.args.val_data, "rb"))
			# self.val_data = raw_data[:val_len]
			# data_len = self.val_data.shape[0]
			# batch_size = 10#self.args.batch_size
			self.val_batch_size = 100#batch_size
			# batch_len = batch_size * self.args.seq_size
			# total_batch = data_len - batch_len + 1
			self.val_total_batch = int(np.ceil(len(self.val_data) / self.val_batch_size))

		if self.args.mode // 2 == 0:
			
			# load testing data
			self.test_data = pickle.load(open(self.args.te_data, "rb"))
			self.te_batch_size = 100
			self.te_total_batch = int(np.ceil(len(self.test_data) / self.te_batch_size))


	# def shuffle(self, data, labels):
	# 	p = np.random.permutation(len(data))
	# 	data = data[p]
	# 	labels = labels[p]
	# 	return data, labels

	# def tmp_data(self, sess, var):

	# 	l = []
	# 	for _ in range(self.tr_total_batch):
	# 		data = sess.run(var['data_input']['train'])
	# 		for d in data:
	# 			if 0 not in d:
	# 				l.append(d)
	# 		# l.append(data)
	# 	self.train_data = np.concatenate(l, axis=1)
	# 	print(self.train_data.shape)

	def next_batch(self, sess, var, dtype="train"):
		if dtype == "train":

			# data = sess.run(var['data_input']['train'])
			
			data = self.train_data[self.index : self.index + self.args.batch_size]
			if self.index + self.args.batch_size >= len(self.train_data):
				self.index = 0
				np.random.shuffle(self.train_data)
			else:
				self.index += self.args.batch_size
			# print(data.shape)
			# x = data
			# x = np.concatenate([data[:,:self.args.seq_size-1], np.zeros((self.args.batch_size, 1))], axis=1)
			x = data[:, :self.args.seq_size - self.args.delay]
			y = data[:, self.args.delay:]
			return x, y
			# data = self.train_data[self.index : self.index + batch_size]
			# labels = self.train_labels[self.index : self.index + batch_size]

			# if self.index + batch_size >= len(self.train_data):
			# 	self.index = 0
			# 	self.train_data, self.train_labels = self.shuffle(self.train_data, self.train_labels)
			# else:
			# 	self.index += batch_size

			# return data, labels 

		elif dtype == "valid":
			# data = sess.run(var['data_input']['valid'])
			
			data = self.val_data[self.index_valid : self.index_valid + self.val_batch_size]
			if self.index_valid + self.val_batch_size >= len(self.val_data):
				self.index_valid = 0
			else:
				self.index_valid += self.val_batch_size

			# x = np.concatenate([data[:,:self.args.seq_size-1], np.zeros((self.val_batch_size, 1))], axis=1)
			# x = data
			x = data[:, :self.args.seq_size - self.args.delay]
			y = data[:, self.args.delay:]
			return x, y
			# data = self.val_data[self.index_valid : self.index_valid + batch_size]
			# labels = self.val_labels[self.index_valid : self.index_valid + batch_size]
			# if self.index_valid + batch_size >= len(self.val_data):
			# 	self.index_valid = 0
			# else:
			# 	self.index_valid += batch_size

			# return data, labels 

		elif dtype == "test":
			
			data = self.test_data[self.index_test : self.index_test + self.te_batch_size]
			if self.index_test + self.te_batch_size >= len(self.test_data):
				self.index_test = 0
			else:
				self.index_test += self.te_batch_size

			# x = np.concatenate([data[:,:self.args.seq_size-1], np.zeros((self.val_batch_size, 1))], axis=1)
			# x = data
			x = data[:, :self.args.seq_size - self.args.delay]
			y = data[:, self.args.delay:]
			return x, y
			# data = self.test_data[self.index_test : self.index_test + batch_size]
			# # labels = self.val_labels[self.index_valid : self.index + batch_size]
			# if self.index_test + batch_size >= len(self.test_data):
			# 	self.index_test = 0
			# else:
			# 	self.index_test += batch_size

			# return data

		

	def add_model(self):


		# input tensor
		data_input = {}
		# stride_num = 1
		# # training input, must shuffle
		# data = tf.convert_to_tensor(self.train_data, dtype=tf.int32)
		# index = tf.train.range_input_producer(self.tr_total_batch, shuffle=True).dequeue()
		# x_seq = tf.strided_slice(data, [index], [index + (self.args.batch_size * self.args.seq_size)], [stride_num])
		# # batch_x = tf.reshape(x_seq, [self.args.batch_size, self.args.seq_size])
		# batch_x = tf.train.maybe_batch([x_seq], tf.reduce_all(tf.cast(x_seq, tf.bool)), 1, num_threads=1, capacity=32, enqueue_many=False, shapes=[self.args.batch_size * self.args.seq_size], dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)
		# data_input['train'] = batch_x

		# # validation input
		# data = tf.convert_to_tensor(self.val_data, dtype=tf.int32)
		# index = tf.train.range_input_producer(self.tr_total_batch, shuffle=False).dequeue()
		# x_seq = tf.strided_slice(data, [index], [index + (self.args.batch_size * self.args.seq_size)], [stride_num])
		# # batch_x = tf.reshape(x_seq, [self.args.batch_size, self.args.seq_size])
		# batch_x = tf.train.maybe_batch([x_seq], tf.reduce_all(tf.cast(x_seq, tf.bool)), 1, num_threads=1, capacity=32, enqueue_many=False, shapes=[self.args.batch_size * self.args.seq_size], dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)
		
		# data_input['val'] = batch_x

		# testing input

		# TODO:

		# model

		# input placeholder
		# x = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_size - self.args.delay])
		# y = tf.placeholder(tf.int64, [self.args.batch_size, self.args.seq_size - self.args.delay])
		x = tf.placeholder(tf.int32, [None, self.args.seq_size - self.args.delay])
		y = tf.placeholder(tf.int64, [None, self.args.seq_size - self.args.delay])
		keep_prob = tf.placeholder(tf.float32)

		hidden_size = self.args.hidden_size

		def lstm_cell_drop(hidden_size):
			# cell = tf.contrib.rnn.BasicLSTMCell(
			# 	hidden_size, forget_bias=0.0, state_is_tuple=True)
			cell = tf.contrib.rnn.LSTMCell(
				hidden_size, use_peepholes=True, forget_bias=0.0, state_is_tuple=True)
			cell_drop = tf.contrib.rnn.DropoutWrapper(
				cell, output_keep_prob=keep_prob)
			return cell_drop

		# hidden_size = 64, num_layers = 2
		cell = tf.contrib.rnn.MultiRNNCell(
			[lstm_cell_drop(hidden_size) for _ in range(self.args.num_layers)], state_is_tuple=True)

		
		embedding = tf.get_variable(
			"embedding", [self.vocab.vocab_size, self.args.dim], dtype=tf.float32)

		embed = tf.nn.embedding_lookup(embedding, x)
		embed_drop = tf.nn.dropout(embed, keep_prob)

		batch_shape = tf.placeholder(tf.int32)
		initial_state = cell.zero_state(batch_shape, dtype=tf.float32)

		outputs = []
		state = initial_state
		with tf.variable_scope("RNN"):
			for step in range(self.args.seq_size - self.args.delay):
				if step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(embed_drop[:, step, :], state)
				outputs.append(cell_output)
		
		# output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
		# output = cell_output

		softmax_w = tf.get_variable(
			"softmax_w", [self.vocab.vocab_size, hidden_size], dtype=tf.float32)
		softmax_b = tf.get_variable(
			"softmax_b", [self.vocab.vocab_size])
		print(outputs[0].get_shape())
		print(y.get_shape())
		losses = []
		for step in range(self.args.seq_size - self.args.delay):
			# if step > 0: tf.get_variable_scope().reuse_variables()
			loss_part = tf.nn.nce_loss(
				softmax_w, softmax_b, 
				y[:,step][:,None], 
				outputs[step], 
				self.args.num_sampled, 
				self.vocab.vocab_size)
			losses.append(loss_part)

		cost = tf.reduce_mean(losses) #/ self.args.batch_size

		logits = []
		labels = []
		softmax_w_t = tf.transpose(softmax_w)
		for step in range(self.args.seq_size - self.args.delay):
			# if step > 0: tf.get_variable_scope().reuse_variables()
			output = outputs[step]
			logit = tf.matmul(output, softmax_w_t) + softmax_b
			# print(logit.get_shape())
			label = y[:, step]
			# print(label.get_shape())
			logits.append(logit)
			labels.append(label)

		v_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			logits, 
			labels,
			[tf.ones([tf.shape(x)[0]], dtype=tf.float32)] * (self.args.seq_size - self.args.delay), 
		)
		print(v_loss.get_shape())
		v_cost = tf.reduce_mean(v_loss)
		# logits = tf.matmul(output, softmax_w) + softmax_b

		# loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
		# 	[logits], 
		# 	[tf.reshape(y, [-1])],
		# 	[tf.ones([self.args.batch_size * (self.args.seq_size - self.args.delay)], 
		# 		dtype=tf.float32 )]
		# 	)
		# print(loss)
		# cost = tf.reduce_sum(loss) / self.args.batch_size
		final_state = state



		# nce_w = tf.get_variable(
		# 	"nce_w", [self.vocab.vocab_size, hidden_size], dtype=tf.float32)
		# nce_b = tf.get_variable(
		# 	"nce_b", [self.vocab.vocab_size], dtype=tf.float32)
		# cost = tf.reduce_mean(
		# 	tf.nn.nce_loss(nce_w, nce_b, y, output, self.args.num_sampled, self.vocab.vocab_size))

		

		# logits = tf.matmul(output, tf.transpose(nce_w))
		# logits = tf.nn.bias_add(logits, nce_b)
		# y_ = tf.argmax(logits, axis=1)

		# correct = tf.equal(y_, y)
		# acc = tf.reduce_mean(tf.cast(correct, tf.float32))

		# labels_one_hot = tf.squeeze(tf.one_hot(y, self.vocab.vocab_size, axis=1))
		# v_cost = tf.nn.sigmoid_cross_entropy_with_logits(
		# 	labels=labels_one_hot,
		# 	logits=logits)
		# print(v_cost.get_shape())
		# v_cost = tf.reduce_mean(tf.reduce_sum(v_cost, 1))


		# logits = tf.matmul(output, softmax_w) + softmax_b
		# print(logits.get_shape())

		# cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
		# 	[logits],
		# 	[tf.reshape(y, [-1])],
		# 	[tf.ones([self.args.batch_size ], dtype=tf.float32)])
		# cost = tf.reduce_sum(cost) / self.args.batch_size



		

		# lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()

		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 
		 						self.args.max_grad_norm)
		optmizer = tf.train.AdagradOptimizer(self.args.lr)

		# optmizer = tf.train.AdagradOptimizer(lr)
		opt = optmizer.apply_gradients(
				zip(grads, tvars))
				# global_step=tf.contrib.framework.get_or_create_global_step())

		# new_lr = tf.placeholder(tf.float32)
		# lr_update = tf.assign(lr, new_lr)

		

		# opt = optmizer.minimize(cost)

		var = {
			'data_input': data_input,
			'embedding': embedding,
			'x': x,
			'y': y,
			'keep_prob': keep_prob,
			'cost': cost, 
			'v_cost': v_cost,
			'v_loss': v_loss,
			# 'y_': y_,
			# 'acc': acc,
			'opt': opt,
			'initial_state': initial_state,
			# 'final_state': final_state,
			# 'new_lr': new_lr,
			# 'lr_update': lr_update
			'batch_shape': batch_shape

		}

		return var

	def train(self):
		with tf.Graph().as_default():
			logging.info("adding model")
			var = self.add_model()

			if self.args.embedding:
				logging.info("loading pre-train embedding")
				embedding = load_bin_vec(self.args.embedding, self.vocab, self.args.dim)


			saver = tf.train.Saver()

			# if self.args.conti:
			# 	sv = ....
			if self.args.embedding:
				sv = tf.train.Supervisor(logdir=self.args.log, saver=saver,
					local_init_op=var['embedding'].assign(embedding))
			else:
				sv = tf.train.Supervisor(logdir=self.args.log, saver=saver)
				
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

			with sv.managed_session(config=config) as sess:

				

				# self.tmp_data(sess, var)

				

				

				
				# sess = tf.Session(config=config)

				

				# sess.run(tf.global_variables_initializer())

				# total_batch = int(np.ceil(len(self.train_data) / float(self.args.batch)))
				total_batch = self.tr_total_batch

				for epoch in range(self.args.epochs):

					# lr_decay = self.args.lr_decay ** max(epoch + 1 - self.args.epochs, 0.0)
					# sess.run(var['lr_update'], feed_dict={var['new_lr']: self.args.lr * lr_decay})
					total_loss = 0.0
					total_acc = 0.0
					total_count = 0
					pbar = pb.ProgressBar(widgets=["[TRAIN] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('acc'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()

					for i in range(total_batch):
						acc = 0
						batchx, batchy = self.next_batch(sess, var)
						_, loss = sess.run([var['opt'], var['cost']], 
							feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:1.0, var['batch_shape']:len(batchx)})
						total_loss += loss 
						total_acc += acc
						total_count += 1.0
						pbar.update(i, loss=total_loss/total_count, acc=total_acc/total_count)
						
					pbar.finish()

					v_loss, v_acc = self.eval(sess, var)
					v_size = self.val_total_batch
					
					logging.info("Epoch {}: tr_loss: {}, tr_acc: {}\n{}v_loss: {}, v_perplex: {}".format(
						epoch, total_loss/total_count, total_acc/total_count, "				   ",
						v_loss/v_size, np.exp(v_loss/v_size)))

				logging.info("saving model")
				save_path = saver.save(sess, self.args.model)
				logging.info("saved model in path: {}".format(save_path))

	def eval(self, sess, var):
		

		total_batch = self.val_total_batch

		total_loss = 0.0
		total_acc = 0.0
		total_count = 0

		pbar = pb.ProgressBar(widgets=["[VALID] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('perplex'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
		for i in range(total_batch):
			batchx, batchy = self.next_batch(sess, var, dtype="valid")
			loss = sess.run(var['v_cost'], 
				feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:1.0, var['batch_shape']:len(batchx)})
			acc = 0
			total_loss += loss 
			total_acc += acc
			total_count += 1.0

			# pbar.update(i)
			pbar.update(i, loss=total_loss/total_count, perplex=np.exp(total_loss/total_count))
		pbar.finish()

		return total_loss, total_acc

	def predict(self):

		total_batch = self.te_total_batch

		with tf.Graph().as_default():

			logging.info("adding model")
			var = self.add_model()

			saver = tf.train.Saver()

			# if self.args.conti:
			# 	sv = ....
			sv = tf.train.Supervisor(logdir=self.args.log, saver=saver)
			
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

			losses = []
			with sv.managed_session(config=config) as sess:

				
				pbar = pb.ProgressBar(widgets=["[TEST] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('perplex'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
				for i in range(total_batch):
					batchx, batchy = self.next_batch(sess, var, dtype="test")
					loss = sess.run(var['v_loss'], 
						feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:1.0, var['batch_shape']:len(batchx)})
					losses.append(loss)
					# print(loss.shape)
					
					pbar.update(i)
				pbar.finish()


				losses = np.concatenate(losses, axis=0).reshape([-1, 5])
				# print(losses.shape)

				# choose less error choices
				choices = np.argmin(losses, axis=1)
				# print(choices.shape)

				answers = ['a', 'b', 'c', 'd', 'e']

				T = [t for t in map(lambda x:answers[x], choices)]

				I = [i for i in range(1, len(T)+1)]

				df = pd.DataFrame({'id':I, 'answer':T}, columns=['id','answer'])

				df.to_csv(self.args.predict, index=False)





				# batchx, batchy = self.next_batch(sess, var, dtype="test")

				# loss = sess.run(var['v_loss'],
				# 		feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:1.0, var['batch_shape']:len(batchx)})
				# print(loss)
				# print(loss.shape)
				

			# var = self.add_model()

			# saver = tf.train.Saver()

			# # config = tf.ConfigProto(allow_soft_placement=True)
			# # config.gpu_options.allow_growth = True
			# # sess = tf.Session(config=config)

			# sess = tf.Session()

			# saver.restore(sess, self.args.model)
			# logging.info("restore model from: {}".format(self.args.model))

			# batch_size = 10000
			# total_batch = int(np.ceil(len(self.test_data) / float(batch_size)))
			# p = open(self.args.predict, "w")
			
			# total_count = 0

			# pbar = pb.ProgressBar(widgets=["[TEST] ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
			# for i in range(total_batch):
			# 	batchx = self.next_batch(batch_size, dtype="test")
			# 	preds = sess.run(var['predict'], feed_dict={var['x']:batchx, var['keep_prob']:1.0})
			# 	fake_indices = range(0,10000)
			# 	answer = np.asarray([fake_indices, preds], dtype=int).T
			# 	np.savetxt(self.args.predict, answer, fmt='%d', header='id,label', delimiter=",", comments='')
			# pbar.finish()
				









if __name__ == "__main__":

	args = arg_parse()

	

	model = RNN(args)

	if args.mode % 2 == 0:
		start_time = time.time()
		model.train()
		logging.info("training time: {}".format(time.time() - start_time))

	if args.mode // 2 == 0:
		start_time = time.time()
		model.predict()
		logging.info("testing time: {}".format(time.time() - start_time))







