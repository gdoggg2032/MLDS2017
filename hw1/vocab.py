from collections import defaultdict, Counter
import logging
logging.basicConfig(level=logging.INFO)
import pickle 
import copy

class Vocab(object):
	def __init__(self, vocab_path=None, vocab_text_list=None):
		self.w2i = {}
		self.i2w = {}
		self.word_count = defaultdict(int)
		self.total_words = 0
		self.vocab_size = 0
		self.unknown = "_UNK"
		self.dummy = "_DUM"
		self.newline = "_NEWLINE"
		self.start = "_START"
		self.end = "_END"
		self.fill = "_____"
		# self.add_word(self.newline, count=0)
		self.add_word(self.dummy, count=0)
		self.add_word(self.unknown, count=0)

		if vocab_text_list:
			self.construct(vocab_text_list)
			pickle.dump(self, open(vocab_path, "wb"))
		elif vocab_path:
			self.__dict__.update(pickle.load(open(vocab_path, "rb")).__dict__)



	def add_word(self, word, count=1):
		if word not in self.w2i:
			index = len(self.w2i)
			self.w2i[word] = index
			self.i2w[index] = word
		self.word_count[word] += count


	def construct(self, words):

		self.cnt = Counter(words)

		for word, count in self.cnt.most_common():
			self.add_word(word, count=count)
		self.total_words = sum(self.word_count.values())
		self.vocab_size = len(self.word_count)
		logging.info("{} total words with vocab size {}".format(self.total_words, self.vocab_size))


	def encode(self, word):
		if word not in self.w2i:
			word = self.unknown
		return self.w2i[word]

	def encode_all(self, words):
		return [self.encode(word) for word in words]

	def encode_all_sents(self, sents):
		return [[self.encode(word) for word in sent] for sent in sents ]

	def decode(self, index):
		assert(index < len(self.word_count)), "Vocab: index out of range"
		return self.i2w[index]

	def decode_all(self, indices):
		return [self.decode(index) for index in indices]

	def decode_all_sents(self, sents):
		return [[self.decode(word) for word in sent] for sent in sents ]

	def __len__(self):
		return len(self.word_count)

