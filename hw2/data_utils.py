import json
import numpy as np
import os
from sample import VideoCaptionSample
from vocab import Vocab 
import re



class VideoCaptionDataManager(object):
	def __init__(self, FLAGS):

		self.total_batch = {'train':0, 'valid':0, 'test':0}
		self.index = {'train':0, 'valid':0, 'test':0}

		if FLAGS.mode % 2 == 0:

			train_data, vocab = load_train_data(
				FLAGS.train_label, 
				FLAGS.train_feat,
				FLAGS.max_step,
				FLAGS.min_count,
				FLAGS.vocab)

			valid_data = load_valid_data(FLAGS.valid_label, FLAGS.valid_feat, vocab)

			self.vocab = vocab 
			self.train_data = train_data
			self.valid_data = valid_data
			self.total_batch['train'] = int(np.ceil(len(self.train_data) / FLAGS.batch_size))
			self.total_batch['valid'] = int(np.ceil(len(self.valid_data) / FLAGS.batch_size))

		if FLAGS.mode // 2 == 0:
			# test_mode
			# self.vocab = ??
			pass


		self.start_id = self.vocab.encode(self.vocab.start)
		self.end_id = self.vocab.encode(self.vocab.end)
		self.num_actions = self.vocab.vocab_size


		

	def draw_batch(self, batch_size, mode='train'):
		if mode == 'train':
			data = self.train_data[self.index['train'] : self.index['train'] + batch_size]
			if self.index['train'] + batch_size >= len(self.train_data):
				self.index['train'] = 0
				np.random.shuffle(self.train_data)
			else:
				self.index['train'] += batch_size
			return data
		if mode == 'valid':
			data = self.valid_data[self.index['valid'] : self.index['valid'] + batch_size]
			if self.index['valid'] + batch_size >= len(self.valid_data):
				self.index['valid'] = 0
			else:
				self.index['valid'] += batch_size
			return data

	def index_to_string(self, indices):
		# remove end_id?
		decoded = self.vocab.decode_sent(indices)
		sent = " ".join(decoded)
		return sent


def clean_string(string):
	return re.sub("[.&!\"',/-]", " ", string)

def build_vocab(captions, max_doc_len, min_count, vocab_path):
	all_text = " ".join([" ".join([clean_string(c) for c in caption['caption']]) for caption in captions]).lower().split()
	vocab = Vocab(max_doc_len, min_count, vocab_path, all_text)
	return vocab


def load_train_data(json_file, feat_dir, max_doc_len, min_count, vocab_path):

	with open(json_file, "r") as f:
		captions = json.load(f)

	
	vocab = build_vocab(captions, max_doc_len, min_count, vocab_path)

	data = []

	for d in captions:
		video_id = d['id']
		video_path = os.path.join(feat_dir, video_id+'.npy')
		image_frames = np.load(video_path)
		# d['frame'] = image_frames
		for c in d['caption']:
			clean_c = clean_string(c).strip().lower().split()
			label_id_list = vocab.encode_sent(clean_c)
			data.append(VideoCaptionSample(video_id, image_frames, label_id_list))

	return data, vocab

def load_valid_data(json_file, feat_dir, vocab):

	with open(json_file, "r") as f:
		captions = json.load(f)

	
	# vocab = build_vocab(captions, max_doc_len, min_count, vocab_path)

	data = []

	for d in captions:
		video_id = d['id']
		video_path = os.path.join(feat_dir, video_id+'.npy')
		image_frames = np.load(video_path)
		# d['frame'] = image_frames
		for c in d['caption']:
			clean_c = clean_string(c).strip().lower().split()
			label_id_list = vocab.encode_sent(clean_c)
			data.append(VideoCaptionSample(video_id, image_frames, label_id_list))

	return data



















