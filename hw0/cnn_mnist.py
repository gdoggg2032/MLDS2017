import tensorflow as tf 
import sys
import struct
import argparse
import time
import numpy as np 
import progressbar as pb
import logging
logging.basicConfig(level=logging.INFO)
from array import array
import csv


def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument("--name", default="cnn", type=str)
	parser.add_argument('--tr_data', default="./train-images-idx3-ubyte", type=str)
	parser.add_argument('--tr_labels', default="./train-labels-idx1-ubyte", type=str)
	parser.add_argument('--te_data', default="./test-image")

	parser.add_argument("--batch", default=100, type=int)
	parser.add_argument("--epochs", default=10, type=int)
	parser.add_argument("--lr", default=1e-2, type=float)

	parser.add_argument("--model", default="./model", type=str)
	parser.add_argument("--predict", default="./predict.csv", type=str)
	parser.add_argument("--mode", default=0, type=int)


	args = parser.parse_args()

	return args 

class CNN(object):
	def __init__(self, args):
		self.args = args
		self.load_data()
		self.index = 0
		self.index_valid = 0
		self.index_test = 0

	def load_data(self):

		logging.info("loading data")

		if args.mode % 2 == 0:
			# load training data
			with open(self.args.tr_labels, "rb") as f:
				magic, size = struct.unpack(">II", f.read(8))
				# assert(magic == 2049, "train_labels magic number error, expect 2049, got {}".format(magic))
				train_labels = np.array(array("B", f.read()))

			with open(self.args.tr_data, "rb") as f:
				magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
				# assert(magic == 2051, "train_data magic number error, expect 2051, got {}".format(magic))
				logging.info("train_data: size: {}, rows: {}, cols: {}".format(size, rows, cols))
				train_data = np.reshape(array("B", f.read()), [size, rows, cols])

			self.train_data = train_data[:, :, :, None] / 255.0
			self.train_labels = train_labels

			# do validation

			per = np.random.permutation(60000)
			valid = per[:1000] # 1/10
			train = per[1000:]
			self.val_data = self.train_data[valid]
			self.val_labels = self.train_labels[valid]
			self.train_data = self.train_data[train]
			self.train_labels = self.train_labels[train]

			# self.val_data = self.train_data
			# self.val_labels = self.train_labels

			


		if args.mode / 2 == 0:
			with open(self.args.te_data, "rb") as f:
				magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
				# assert(magic == 2051, "test_data magic number error, expect 2051, got {}".format(magic))
				logging.info("test_data: size: {}, rows: {}, cols: {}".format(size, rows, cols))
				test_data = np.reshape(array("B", f.read()), [size, rows, cols])

			self.test_data = test_data[:, :, :, None] / 255.0

	def shuffle(self, data, labels):
		p = np.random.permutation(len(data))
		data = data[p]
		labels = labels[p]
		return data, labels

	def next_batch(self, batch_size, dtype="train"):
		if dtype == "train":
			data = self.train_data[self.index : self.index + batch_size]
			labels = self.train_labels[self.index : self.index + batch_size]

			if self.index + batch_size >= len(self.train_data):
				self.index = 0
				self.train_data, self.train_labels = self.shuffle(self.train_data, self.train_labels)
			else:
				self.index += batch_size

			return data, labels 

		elif dtype == "valid":
			data = self.val_data[self.index_valid : self.index_valid + batch_size]
			labels = self.val_labels[self.index_valid : self.index_valid + batch_size]
			if self.index_valid + batch_size >= len(self.val_data):
				self.index_valid = 0
			else:
				self.index_valid += batch_size

			return data, labels 

		elif dtype == "test":
			data = self.test_data[self.index_test : self.index_test + batch_size]
			# labels = self.val_labels[self.index_valid : self.index + batch_size]
			if self.index_test + batch_size >= len(self.test_data):
				self.index_test = 0
			else:
				self.index_test += batch_size

			return data

		

	def add_model(self):
		def weight_variable(shape, dev):
			initial = tf.random_normal(shape, stddev=dev)
			return tf.Variable(initial)
		def bias_variable(shape):
			initial = tf.random_normal(shape)
			return tf.Variable(initial)
		def conv2d(x_in, W):
			return tf.nn.conv2d(x_in, W, 
				strides=[1,1,1,1], padding='SAME')
		def max_pool_2x2(x_in):
			return tf.nn.max_pool(x_in, ksize=[1,2,2,1],
				strides=[1,2,2,1], padding='SAME')
		def activation(x_in, activ):
			if activ == "relu":
				return tf.nn.relu(x_in)
			elif activ == "linear":
				return x_in
			elif activ == "tanh":
				return tf.nn.tanh(x_in)

		x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
		y = tf.placeholder(tf.int64, shape=[None])
		keep_prob = tf.placeholder(tf.float32)

		def conv_layer(x_in, w_shape, b_shape, activ="relu"):
			W_conv = weight_variable(w_shape, 0.01)
			b_conv = bias_variable(b_shape)
			a_conv = conv2d(x_in, W_conv) + b_conv
			h_conv = activation(a_conv, activ)
			h_pool = max_pool_2x2(h_conv)
			h_pool_drop = tf.nn.dropout(h_pool, keep_prob)

			return h_pool_drop

		h1_pool_drop = conv_layer(x, [3, 3, 1, 64], [64])
		h2_pool_drop = conv_layer(h1_pool_drop, [3, 3, 64, 128], [128])
		h3_pool_drop = conv_layer(h2_pool_drop, [3, 3, 128, 256], [256])

		h_flat = tf.reshape(h3_pool_drop, [-1, 4 * 4 * 256])

		def fc_layer(x_in, w_shape, b_shape, activ="relu"):
			W_fc = weight_variable(w_shape, 0.01)
			b_fc = bias_variable(b_shape)
			a_fc = tf.matmul(x_in, W_fc) + b_fc
			h_fc = activation(a_fc, activ)
			h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

			return h_fc_drop

		h1_fc_drop = fc_layer(h_flat, [4 * 4 * 256, 1024], [1024])
		h2_fc_drop = fc_layer(h1_fc_drop, [1024, 1024], [1024])
		h3_fc_drop = fc_layer(h2_fc_drop, [1024, 10], [10], "linear")
		y_ = h3_fc_drop

		cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(y_, y))
		optimizer = tf.train.AdagradOptimizer(self.args.lr)
		opt = optimizer.minimize(cost)

		predict = tf.argmax(y_, 1)
		correct = tf.equal(predict, y)
		acc_sum = tf.reduce_sum(tf.cast(correct, tf.float32))

		var = {
			"x":x,
			"y":y,
			"keep_prob":keep_prob,
			"opt":opt,
			"cost":cost,
			"acc_sum":acc_sum,
			"predict":predict,
		}

		return var

	def train(self):
		with tf.Graph().as_default():

			logging.info("add model")

			var = self.add_model()

			saver = tf.train.Saver()

			# config = tf.ConfigProto(allow_soft_placement=True)
			# config.gpu_options.allow_growth = True
			# sess = tf.Session(config=config)

			sess = tf.Session()

			sess.run(tf.initialize_all_variables())

			total_batch = int(np.ceil(len(self.train_data) / float(self.args.batch)))

			for epoch in xrange(self.args.epochs):

				total_loss = 0.0
				total_acc_sum = 0.0
				total_count = 0
				pbar = pb.ProgressBar(widgets=["[TRAIN] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('acc'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()

				for i in xrange(total_batch):
					batchx, batchy = self.next_batch(self.args.batch)
					_, loss, acc_sum = sess.run([var['opt'], var['cost'], var['acc_sum']], 
						feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:0.7})
					total_loss += loss 
					total_acc_sum += acc_sum
					total_count += len(batchx)
					pbar.update(i, loss=total_loss/total_count, acc=total_acc_sum/total_count)
				pbar.finish()

				v_loss, v_acc_sum = self.eval(sess, var)
				v_size = len(self.val_data)
				
				logging.info("Epoch {}: tr_loss: {}, tr_acc: {}\n{}v_loss: {}, v_acc: {}".format(
					epoch, total_loss/total_count, total_acc_sum/total_count, "                   ",
					v_loss/v_size, v_acc_sum/v_size))

			logging.info("save model")
			save_path = saver.save(sess, self.args.model)
			logging.info("save model in path: {}".format(save_path))

	def eval(self, sess, var):
		batch_size = 200

		total_batch = int(np.ceil(len(self.val_data) / float(batch_size)))

		total_loss = 0.0
		total_acc_sum = 0.0
		total_count = 0

		pbar = pb.ProgressBar(widgets=["[VALID] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('acc'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
		for i in xrange(total_batch):
			batchx, batchy = self.next_batch(batch_size, dtype="valid")
			loss, acc_sum = sess.run([var['cost'], var['acc_sum']], 
				feed_dict={var['x']:batchx, var['y']:batchy, var['keep_prob']:1.0})
			total_loss += loss 
			total_acc_sum += acc_sum
			total_count += len(batchx)
			pbar.update(i, loss=total_loss/total_count, acc=total_acc_sum/total_count)
		pbar.finish()

		return total_loss, total_acc_sum

	def predict(self):
		with tf.Graph().as_default():
			var = self.add_model()

			saver = tf.train.Saver()

			# config = tf.ConfigProto(allow_soft_placement=True)
			# config.gpu_options.allow_growth = True
			# sess = tf.Session(config=config)

			sess = tf.Session()

			saver.restore(sess, self.args.model)
			logging.info("restore model from: {}".format(self.args.model))

			batch_size = 10000
			total_batch = int(np.ceil(len(self.test_data) / float(batch_size)))
			p = open(self.args.predict, "w")
			
			total_count = 0

			pbar = pb.ProgressBar(widgets=["[TEST] ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
			for i in xrange(total_batch):
				batchx = self.next_batch(batch_size, dtype="test")
				preds = sess.run(var['predict'], feed_dict={var['x']:batchx, var['keep_prob']:1.0})
				fake_indices = range(0,10000)
				answer = np.asarray([fake_indices, preds], dtype=int).T
				np.savetxt(self.args.predict, answer, fmt='%d', header='id,label', delimiter=",", comments='')
			pbar.finish()
				









if __name__ == "__main__":

	args = arg_parse()

	

	model = CNN(args)

	if args.mode % 2 == 0:
		start_time = time.time()
		model.train()
		logging.info("training time: {}".format(time.time() - start_time))

	if args.mode / 2 == 0:
		start_time = time.time()
		model.predict()
		logging.info("testing time: {}".format(time.time() - start_time))







