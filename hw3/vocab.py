from collections import defaultdict, Counter
import logging
logging.basicConfig(level=logging.INFO)
import pickle 
import copy

class Vocab(object):
	def __init__(self, min_count=5, vocab_path=None, vocab_text_list=None):
		self.w2i = {}
		self.i2w = {}
		self.word_count = defaultdict(int)
		self.total_words = 0
		self.vocab_size = 0
		self.unknown = "_UNK_"
		self.dummy = "<EOS>" #"_DUM_" 
		# use EOS for easy to compute score
		self.start = "<BOS>"
		self.end = "<EOS>"
		# self.add_word(self.newline, count=0)
		self.add_word(self.unknown, count=0)
		self.min_count = min_count

		if vocab_text_list:
			self.construct(vocab_text_list, min_count)
			pickle.dump(self, open(vocab_path, "wb"))
		elif vocab_path:
			self.__dict__.update(pickle.load(open(vocab_path, "rb")).__dict__)

	def dump(self, vocab_path):
		pickle.dump(self, open(vocab_path, "wb"))

	def add_word(self, word, count=1):
		if word not in self.w2i:
			index = len(self.w2i)
			self.w2i[word] = index
			self.i2w[index] = word
			self.vocab_size += 1
		self.word_count[word] += count


	def construct(self, words, min_count):

		self.cnt = Counter(words)

		for word, count in self.cnt.most_common():
			if count >= min_count:
				self.add_word(word, count=count)
		self.total_words = sum(self.word_count.values())
		self.vocab_size = len(self.word_count)
		logging.info("{} total words with vocab size {}".format(self.total_words, self.vocab_size))


	def encode(self, word):
		if word not in self.w2i:
			word = self.unknown
		return self.w2i[word]

	def decode(self, index):
		assert(index < len(self.word_count)), "Vocab: index out of range"
		return self.i2w[index]


	def __len__(self):
		return len(self.word_count)

