import logging
logging.basicConfig(level=logging.INFO)

from vocab import Vocab 
import os 
import time
import random
import numpy as np 

import re

class Data(object):
	def __init__(self, state, action, reward, prev_target, target):
		self.state = state 
		self.action = action 
		self.reward = reward
		self.prev_target = prev_target
		self.target = target




class DataManager(object):

	def __init__(self,
				mode,
				num_step,
				train_file_path=None,
				test_file_path=None,
				vocab_path=None,
				min_count=10):

		self.index = {'train':0, 'test':0}

		self.num_step = num_step

		if mode % 2 == 0:
			s_time = time.time()
			self.train_data, self.vocab = self.load_train_data(train_file_path, min_count)
			logging.info("train_data len: {}".format(len(self.train_data)))
			logging.info("load train_data time: {}".format(time.time() - s_time))
			logging.info("vocab size: {}".format(self.vocab.vocab_size))
			self.vocab.dump(vocab_path)

			self.replay_data = []

		if mode // 2 == 0:
			self.vocab = Vocab(vocab_path=vocab_path)
			self.test_data = self.load_test_data(test_file_path, self.vocab)

	def clear_sent(self, string):
		return re.sub("[^a-z]", " ", string.strip().lower())

	def load_train_data(self, train_file_path, min_count):

		# assume every two line as a data
		vocab = Vocab(min_count=min_count)
		with open(train_file_path, "r") as f:
			D = self.clear_sent(f.read()).split()
			vocab.construct(D)

		data = []
		with open(train_file_path, "r") as f:
			for line1, line2 in zip(f, f):

				line1 = self.clear_sent(line1).split()
				line2 = self.clear_sent(line2).split()

				if len(line1) < 1 or len(line2) < 1:
					continue

				if len(line1) > self.num_step - 2 or len(line2) > self.num_step - 2 + 1:
					continue

				line1_encoded = [vocab.encode(vocab.start)]
				line2_encoded = [vocab.encode(vocab.start)]

				# add to vocab and encode
				for w in line1:
					# vocab.add_word(w)
					line1_encoded.append(vocab.encode(w))
					if len(line1_encoded) == self.num_step - 1:
						break
				while len(line1_encoded) < self.num_step:
					line1_encoded.append(vocab.encode(vocab.end))

				# line2 is 1 more step for shift decoder input and output
				for w in line2:
					# vocab.add_word(w)
					line2_encoded.append(vocab.encode(w))
					if len(line2_encoded) == self.num_step:
						break
				while len(line2_encoded) < self.num_step + 1:
					line2_encoded.append(vocab.encode(vocab.end))


				# create instance
				data.append(Data(line1_encoded, None, None, line2_encoded[:-1], line2_encoded[1:]))

		return data, vocab 


	def load_test_data(self, test_file_path, vocab):
		data = []
		with open(test_file_path, "r") as f: 
			for line in f:
				line = self.clear_sent(line).split()


				line_encoded = [vocab.encode(vocab.start)]

				# add to vocab and encode
				for w in line:
					# vocab.add_word(w)
					line_encoded.append(vocab.encode(w))
					if len(line_encoded) == self.num_step - 1:
						break
				while len(line_encoded) < self.num_step:
					line_encoded.append(vocab.encode(vocab.end))

				data.append(Data(line_encoded, None, None, None, None))
		return data 


	def draw_batch(self, batch_size, mode='train'):
		
		if mode == 'train':
			data = self.train_data[self.index['train'] : self.index['train'] + batch_size]
			if self.index['train'] + batch_size >= len(self.train_data):
				self.index['train'] = 0
				np.random.shuffle(self.train_data)
			else:
				self.index['train'] += batch_size

			return data 

		if mode == 'random':
			data = random.sample(self.train_data, batch_size)
			return data

		if mode == 'test':
			data = self.test_data[self.index['test'] : self.index['test'] + batch_size]
			self.index['test'] += batch_size

			return data

	def total_batch_num(self, batch_size, mode='train'):

		if mode == 'train':
			return int(np.ceil(len(self.train_data) / batch_size))
	
		if mode == 'test':
			return int(np.ceil(len(self.test_data) / batch_size))













			