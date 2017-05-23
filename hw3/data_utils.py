import numpy as np 
import skimage
import skimage.io
import skimage.transform


import logging
logging.basicConfig(level=logging.INFO)

from vocab import Vocab 

import csv
import os 

import random 
import time

import scipy.stats as stats 

import re



class Data(object):
	def __init__(self, img_id, img, tags, tag_text, with_text=1):
		self.img_id = img_id
		self.img = img 
		self.tags = tags 
		self.tag_text = tag_text
		self.with_text = with_text


def quantize(image, L=1, N=4):
	"""Quantize an image.
	Parameters
	----------
	image : array_like
		Input image.
	L : float
		Maximum input value.
	N : int
		Number of quantization levels.
	"""
	T = np.linspace(0, L, N, endpoint=False)[1:]
	return np.digitize(image.flat, T).reshape(image.shape)


class Sampler(object):
	def __init__(self, z_type):
		self.z_type = z_type
		if self.z_type == 'truncnorm':
			self.sampler = stats.truncnorm(-1, 1, loc=0, scale=1)
		

	def sample(self, batch_size, z_dim):
		if self.z_type == 'truncnorm':
			return self.sampler.rvs([batch_size, z_dim])
		elif self.z_type == 'uniform':
			return np.random.uniform(-1, 1, size=[batch_size, z_dim])
		elif self.z_type == 'normal':
			return np.random.normal(0, 1, size=[batch_size, z_dim])


class DataManager(object):
	def __init__(self,
				 mode,
				 tag_file_path=None,
				 img_dir_path=None,
				 test_text_path=None,
				 vocab_path=None,
				 z_dim=100,
				 z_type='normal',
				 generator_output_layer='tanh'):

		self.index = {'train':0, 'test':0}

		self.tag_num = 2

		self.label_data = {}
		self.nega_data = {}

		self.z_sampler = Sampler(z_type=z_type)

		self.unk_counter = 0

		self.generator_output_layer = generator_output_layer

		if mode % 2 == 0:
			s_time = time.time()
			self.train_data, self.vocab = self.load_train_data(tag_file_path, img_dir_path)
			logging.info("train_data len: {}".format(len(self.train_data)))
			logging.info("with_unk train data len: {}".format(self.unk_counter))
			logging.info("load train_data time: {}".format(time.time() - s_time))
			self.vocab.dump(vocab_path)
			# self.count_priority()
			s_time = time.time()
			self.find_negatives()
			logging.info("find negatives time: {}".format(time.time() - s_time))


			# choose 10 images for gif
			# each tag sample one image for tracking
			self.gif_data = []
			for key in self.label_data:
				d = random.sample(self.label_data[key], 1)[0]
				self.gif_data.append(d)
			self.gif_z = self.z_sampler.sample(len(self.gif_data), z_dim)
			


		if mode // 2 == 0:
			self.vocab = Vocab(vocab_path=vocab_path)
			self.test_data = self.load_test_data(test_text_path, self.vocab)

	def load_train_data(self, tag_file_path, img_dir_path):

		vocab = Vocab(min_count=0)
		for color in ['blonde', 'brown', 'black', 'blue', 'pink', 'purple', 'red', 
					  'green', 'gray', 'aqua', 'white', 'orange',
					  'yellow', 'bicolored']:
			vocab.add_word(color)

		# train data analysis

		# [('long hair', 8463),
		#  ('short hair', 4061),
		#  ('blonde hair', 3735),
		#  ('brown hair', 3609),
		#  ('black hair', 2709),
		#  ('blue hair', 1875),
		#  ('pink hair', 1781),
		#  ('purple hair', 1346),
		#  ('red hair', 1053),
		#  ('green hair', 1016),
		#  ('gray hair', 926),
		#  ('aqua hair', 877),
		#  ('white hair', 871),
		#  ('orange hair', 410),
		#  ('damage hair', 1),
		#  ('pubic hair', 1)]

		# [('blue eyes', 4025),
		#  ('red eyes', 3025),
		#  ('brown eyes', 2118),
		#  ('green eyes', 1988),
		#  ('purple eyes', 1739),
		#  ('aqua eyes', 1290),
		#  ('yellow eyes', 1193),
		#  ('pink eyes', 691),
		#  ('orange eyes', 362),
		#  ('black eyes', 334),
		#  ('bicolored eyes', 307),
		#  ('gray eyes', 182),
		#  ('11 eyes', 12)]

		data = []


		with open(tag_file_path, "r") as tag_f:
			reader = csv.reader(tag_f)
			for img_id, tag_str in reader:
				# convert img_id to integer
				img_id = int(img_id)

				# tag_str
				# my policy: only put hair color and eye color
				# cut all :integer
				tags = [s.split(":")[0].strip() for s in tag_str.lower().split("\t")]
				# filter all not hair and not eyes
				hair = [t.split(" ")[0] for t in tags if t.endswith('hair')]
				eyes = [t.split(" ")[0] for t in tags if t.endswith('eyes')]

				# filter all not color
				hair = [vocab.encode(h) for h in hair if h in vocab.w2i and vocab.encode(h) != vocab.unknown]
				eyes = [vocab.encode(e) for e in eyes if e in vocab.w2i and vocab.encode(e) != vocab.unknown]


				# skip no eye hair tag data
				if len(hair) == 0 and len(eyes) == 0:
					continue

				# skip > 1 hair or >1 eyes, because they are someone not shown in images
				if len(hair) > 1 or len(eyes) > 1:
					continue

				# if len(hair) == 0 or len(hair) > 1:
				# 	# hair = [vocab.encode(vocab.unknown)]
				# 	hair = []
				# if len(eyes) == 0 or len(eyes) > 1:
				# 	# eyes = [vocab.encode(vocab.unknown)]
				# 	eyes = []


				with_text = 1
				with_unk = 0

				if len(hair) == 0 or len(hair) > 1 or len(eyes) == 0 or len(eyes) > 1:

					if len(hair) == 1:
						eyes = [vocab.encode(vocab.unknown)]
						with_unk = 1

					elif len(eyes) == 1:
						hair = [vocab.encode(vocab.unknown)]
						with_unk = 1

					else:
						hair = []
						eyes = []
						with_text = 0
						with_unk = 1


					# if (len(hair) == 0 and len(eyes) == 0) or (len(hair) > 1 and len(eyes) > 1):
					# 	hair = []
					# 	eyes = []
					# 	with_text = 0
					# else:
					# 	if len(hair) == 0 or len(hair) > 1:
					# 		hair = [vocab.encode(vocab.unknown)]
					# 	if len(eyes) == 0 or len(eyes) > 1:
					# 		eyes = [vocab.encode(vocab.unknown)]

				hair_str = [vocab.decode(h) for h in hair]
				eyes_str = [vocab.decode(e) for e in eyes]
				tag_text = "{}_hair_{}_eyes".format("_".join(hair_str), "_".join(eyes_str))

		

				hair = set(hair)
				eyes = set(eyes)
				# hair dim concat eyes dim
				# bag of words
				feature = np.zeros((self.tag_num * vocab.vocab_size))

				for c_id in hair:
					# if c_id == vocab.encode(vocab.unknown):
					# 	feature[0:vocab.vocab_size] += 0#1 / vocab.vocab_size
					# else:
					feature[c_id] += 1
				for c_id in eyes:
					# if c_id == vocab.encode(vocab.unknown):
					# 	feature[vocab.vocab_size:] += 0#1 / vocab.vocab_size
					# else:
					feature[c_id + vocab.vocab_size] += 1

				# image
				img_path = os.path.join(img_dir_path, str(img_id) + ".jpg")

				# load image
				# convert img to -1, 1
				if self.generator_output_layer == 'tanh':
					img = skimage.io.imread(img_path) / 127.5 - 1
				elif self.generator_output_layer == 'sigmoid':
					img = skimage.io.imread(img_path) / 255

				# img = skimage.io.imread(img_path)
				# img = quantize(img, L=255, N=10)
				# img = img / 5 - 1


				# resize to 64 * 64
				img_resized = skimage.transform.resize(img, (64, 64), mode='constant')
				# try: do not resize
				# img_resized = img 

				no_text = "{}_hair_{}_eyes".format('', '')

				if tag_text == no_text:
					feature = np.zeros((self.tag_num * vocab.vocab_size)) / (vocab.vocab_size)#np.ones((self.tag_num * vocab.vocab_size)) / (vocab.vocab_size)


				# add rotate image and horizon fliped image
				for angle in [-20, -10, 0, 10, 20]:

					img_rotated = skimage.transform.rotate(img_resized, angle, mode='edge')

					for flip in [0, 1]:

						if flip:
							d = Data(img_id, np.fliplr(img_rotated), feature, tag_text, with_text)
						else:
							d = Data(img_id, img_rotated, feature, tag_text, with_text)


						if tag_text not in self.label_data:
							self.label_data[tag_text] = []


						if with_text:
							self.label_data[tag_text].append(d)

						if with_unk:
							self.unk_counter += 1

						data.append(d)

		return data, vocab

	def parse_tag_text(self, tag_text):
		hair_str = re.findall('.*(?=_hair_)', tag_text)[0]
		eyes_str = re.findall('(?<=_hair_){1}.*(?=_eyes)', tag_text)[0]
		return hair_str, eyes_str

	def find_negatives(self):

		for tag_text1 in self.label_data:
			for tag_text2 in self.label_data:
				if tag_text1 != tag_text2:
					hair_str1, eyes_str1 = self.parse_tag_text(tag_text1)
					hair_str2, eyes_str2 = self.parse_tag_text(tag_text2)
					if self.vocab.unknown in [hair_str1, hair_str2]:
						if self.vocab.unknown not in [eyes_str1, eyes_str2] and eyes_str1 != eyes_str2:
							if tag_text2 not in self.nega_data:
								self.nega_data[tag_text2] = []
							self.nega_data[tag_text2].extend(self.label_data[tag_text1])
					elif self.vocab.unknown in [eyes_str1, eyes_str2]:
						if self.vocab.unknown not in [hair_str1, hair_str2] and hair_str1 != hair_str2:
							if tag_text2 not in self.nega_data:
								self.nega_data[tag_text2] = []
							self.nega_data[tag_text2].extend(self.label_data[tag_text1])
					else:
						if tag_text2 not in self.nega_data:
								self.nega_data[tag_text2] = []
						self.nega_data[tag_text2].extend(self.label_data[tag_text1])


		# text = 'blonde_hair_blue_eyes'
		# for d in self.nega_data[text]:
		# 	hair_str, eyes_str = self.parse_tag_text(d.tag_text)
		# 	if hair_str == 'blonde' and eyes_str == 'blue':
		# 		print("error", text, d.tag_text)

		# text = 'purple_hair__UNK__eyes'
		# for d in self.nega_data[text]:
		# 	hair_str, eyes_str = self.parse_tag_text(d.tag_text)
		# 	if hair_str == 'purple':
		# 		print("error", text, d.tag_text)



		# no_text = "{}_hair_{}_eyes".format('', '')



		# for tag_text1 in self.label_data:
		# 	for tag_text2 in self.label_data:
		# 		if tag_text1 != tag_text2 and tag_text1 != no_text and tag_text2 != no_text:
		# 			if tag_text2 not in self.nega_data:
		# 				self.nega_data[tag_text2] = []
		# 			if self.vocab.unknown not in tag_text1 and self.vocab.unknown not in tag_text2:
		# 				# no unk
		# 				self.nega_data[tag_text2].extend(self.label_data[tag_text1])
		# 			elif self.vocab.unknown not in tag_text1:
		# 				# tag_text2 has unk
		# 				if "{}_hair".format(self.vocab.unknown) in tag_text2:
		# 					# tag_text2 is unk_hair


		# self.nega_data[no_text] = []

		# for _ in range(1000):

		# 	h_noise = np.random.uniform(0, 1, size=(self.tag_num * self.vocab.vocab_size))
		# 	h_noise[:self.vocab.vocab_size] /= np.sum(h_noise[:self.vocab.vocab_size])
		# 	h_noise[self.vocab.vocab_size:] /= np.sum(h_noise[self.vocab.vocab_size:])
		# 	d = Data(-1, np.random.uniform(-1, 1, size=(64, 64, 3)), h_noise, "dummy", 0)

		# 	self.nega_data[no_text].append(d)



		# for i in range(len(self.train_data)):
		# 	for j in range(i+1, len(self.train_data)):
		# 		if not np.array_equal(self.train_data[i].tags, self.train_data[j].tags):
		# 			self.train_data[i].nega_data.append(self.train_data[j])
		# 			self.train_data[j].nega_data.append(self.train_data[i])

	# def count_priority(self):
	# 	feat = [np.matmul(d.tags[:15, None], d.tags[None, 15:]).reshape(-1, 15 * 15) for d in self.train_data]
	# 	feat = np.concatenate(feat, axis=0)
	# 	feat_normal = np.sum(feat, axis=0) / np.sum(feat)


	# 	self.priority = np.matmul(feat, feat_normal)

	# 	self.priority = self.priority / sum(self.priority)



	def load_test_data(self, test_text_path, vocab):
		data = []

		with open(test_text_path, "r") as f:
			reader = csv.reader(f)
			for text_id, text in reader:
				text_id = int(text_id)

				# text format: color hair color eyes
				text_list = text.lower().split(" ")
				hair_color_id = vocab.encode(text_list[0])
				eyes_color_id = vocab.encode(text_list[2])

				feature = np.zeros((self.tag_num * vocab.vocab_size))

				feature[hair_color_id] += 1
				feature[eyes_color_id + vocab.vocab_size] += 1

				for img_id in range(1, 5+1):
					data.append(Data("{}_{}".format(text_id, img_id), None, feature, text.lower().replace(" ", "_")))

		return data

	def draw_batch(self, batch_size, z_dim, mode='train'):
		if mode == 'train':
			data = self.train_data[self.index['train'] : self.index['train'] + batch_size]
			if self.index['train'] + batch_size >= len(self.train_data):
				self.index['train'] = 0
				np.random.shuffle(self.train_data)
			else:
				self.index['train'] += batch_size
			# noise = np.random.normal(size=(len(data), z_dim))
			noise = self.z_sampler.sample(len(data), z_dim)

			# noise_h and noise image
			noise_h = []
			wrong_img = []
			for d in data:
				nega_d = random.sample(self.nega_data[d.tag_text], 1)[0]
				noise_h.append(nega_d.tags)
				wrong_img.append(nega_d.img)


			# noise_h = np.random.multinomial(1, np.ones(self.vocab.vocab_size)/self.vocab.vocab_size,
			#  (len(data), self.tag_num)).reshape(-1, self.tag_num * self.vocab.vocab_size)
			
			# wrong_data = random.sample(self.train_data, len(data))

			return data, noise, noise_h, wrong_img

		# if mode == 'train_priority':
		# 	data = np.random.choice(self.train_data, size=batch_size, p=self.priority)
		# 	noise = np.random.normal(size=(len(data), z_dim))
		# 	noise_h = np.random.multinomial(1, np.ones(self.vocab.vocab_size)/self.vocab.vocab_size,
		# 	 (len(data), self.tag_num)).reshape(-1, self.tag_num * self.vocab.vocab_size)
		# 	wrong_data = random.sample(self.train_data, len(data))

		# 	# # fully negative
		# 	# noise_h = []
		# 	# for i in range(len(data)):
		# 	# 	p = (np.ones(self.vocab.vocab_size ** 2) - data[i].tags ) / (self.vocab.vocab_size ** 2 - 1)
		# 	# 	hair_p = (np.ones(15) - data[i].tags[:15]) / (15 - 1)
		# 	# 	eyes_p = (np.ones(15) - data[i].tags[15:]) / (15 - 1)
		# 	# 	nega_hair = np.random.multinomial(1, hair_p, (1))
		# 	# 	nega_eyes = np.random.multinomial(1, eyes_p, (1))
		# 	# 	nega = np.concatenate([nega_hair, nega_eyes], axis=1)
		# 	# 	nega = np.random.multinomial(1, p, (1))
		# 	# 	noise_h.append(nega)
		# 	# noise_h = np.concatenate(noise_h, axis=0)

		# 	return data, noise, noise_h, wrong_data

		if mode == 'test':
			data = self.test_data[self.index['test'] : self.index['test'] + batch_size]
			if self.index['test'] + batch_size >= len(self.test_data):
				self.index['test'] = 0
			else:
				self.index['test'] += batch_size
			noise = self.z_sampler.sample(len(data), z_dim)
			return data, noise

		if mode == 'random':
			data = random.sample(self.train_data, batch_size)
			noise = self.z_sampler.sample(len(data), z_dim)
			return data, noise

		if mode == 'gif':
			data = self.gif_data
			noise = self.gif_z
			return data, noise

	def total_batch_num(self, batch_size, mode='train'):

		if mode == 'train':
			return int(np.ceil(len(self.train_data) / batch_size))
	
		if mode == 'test':
			return int(np.ceil(len(self.test_data) / batch_size))

def make_gif(valid_img_dir, gif_dir, duration=0.5, true_image=True):

	import re
	import moviepy.editor as mpy

	gifs = {}

	for filename in os.listdir(valid_img_dir):
		if filename.startswith('GIF'):
			# print(filename)

			gif_name = "_".join(filename.split("_")[1:-1])
			if gif_name not in gifs:
				gifs[gif_name] = []
			image = skimage.io.imread(os.path.join(valid_img_dir, filename))
	
			img_seq_id = int(re.search("(?<=_)[0-9]*(?=.png)", filename).group(0))
			gifs[gif_name].append((img_seq_id, image))

	for k in gifs:
		gifs[k] = [i[1] for i in sorted(gifs[k], key=lambda x:x[0])]
		images = gifs[k]

		def make_frame(t):
			try:
				x = images[int(len(images)/duration*t)]
			except:
			 	x = images[-1]

			if true_image:
				return x.astype(np.uint8)
			else:
				return (x*255).astype(np.uint8)

		gif_name = os.path.join(gif_dir, "{}.gif".format(k))

		clip = mpy.VideoClip(make_frame, duration=duration)
		clip.write_gif(gif_name, fps=len(images)/duration, loop=1)
		
if __name__ == "__main__":
	if not os.path.exists("./gif_samples"):
		os.makedirs("./gif_samples")
	make_gif("./valid_samples", "./gif_samples")


	






