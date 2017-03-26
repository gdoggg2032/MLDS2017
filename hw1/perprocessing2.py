import sys
import argparse
import time
import logging
logging.basicConfig(level=logging.INFO)
from collections import Counter, deque
import re
import progressbar as pb

from vocab import Vocab
import numpy as np 
import pickle

import nltk



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument("--name", default="rnnlm_preprocessing", type=str)
	parser.add_argument('--raw_tr_data', default="./all.txt", type=str)
	parser.add_argument('--tr_data', default="./tr_data.pickle", type=str)
	parser.add_argument('--val_data', default="./val_data.pickle", type=str)
	parser.add_argument('--raw_te_data', default="./testing_data.csv", type=str)
	parser.add_argument('--te_data', default="./te_data.pickle", type=str)
	# parser.add_argument('--tr_data_parsed', default="./all_parsed.txt", type=str)

	parser.add_argument('--min_count', default=100, type=int)
	parser.add_argument("--seq_size", default=40, type=int)

	parser.add_argument('--vocab', default="./vocab", type=str)

	args = parser.parse_args()

	return args 

def preprocessing(args):

	vocab = Vocab()

	stemmer = nltk.porter.PorterStemmer()


	text = open(args.raw_tr_data, "r", errors="ignore").read().strip()
	logging.info("reserve alphabet")

	regex = re.compile('[^a-zA-Z \n.!?,*;]')
	text_clear = regex.sub(' ', text).lower()
	# regex = re.compile('[\n.]')
	# text_clear = regex.sub(' _NEWLINE ', text_clear)
	text_clear = text_clear.replace("\n\n", " _NEWLINE ")
	text_clear = text_clear.replace(". ", " _NEWLINE ")
	text_clear = text_clear.replace(".\n", " _NEWLINE ")
	text_clear = text_clear.replace("!", " _NEWLINE ")
	text_clear = text_clear.replace("?", " _NEWLINE ")
	text_clear = text_clear.replace("*\n", " _NEWLINE ")
	text_clear = text_clear.replace(";", " _NEWLINE ")
	text_clear = re.sub("\n(?=[A-Z])", " _NEWLINE ", text_clear)
	text_clear = re.sub("\*(.*)\*", " ", text_clear)
	text_clear = text_clear.replace("\n", " ")
	text_clear = text_clear.replace(".", "")
	text_clear = text_clear.replace(",", " , ")
	text_clear = text_clear.replace("*", " ")

	text_list = text_clear.split("_NEWLINE")

	corpus = []
	text_stem_sent_list = []

	for sent in text_list:
		t = []
		s_l = sent.split()
		for word in s_l:
			word = stemmer.stem(word)
			corpus.append(word)
			t.append(word)
		text_stem_sent_list.append(t)

	logging.info("build counter")
	cnt = Counter(corpus)

	filtered_list = []


	pbar = pb.ProgressBar(widgets=["[PREPROCESS] ", pb.DynamicMessage('sent_len'), " ", pb.DynamicMessage('dummy'), " ", pb.FileTransferSpeed(unit="sents"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(text_list)).start()
	
	val_len = len(filtered_list) // 10

	val_data = []
	train_data = []
	corpus = []

	for i, sent in enumerate(text_stem_sent_list):

		t = deque(maxlen=args.seq_size)
		sent = [vocab.start] + sent + [vocab.end]

		for word in sent:
			if word == "" or word == " ":
				continue
			if cnt[word] < args.min_count:
				word = vocab.unknown
			t.append(word)
			if len(t) == args.seq_size:
				filtered_list.append(list(t))

		if len(t) >= 5:
			while len(t) < args.seq_size:
				t.append(vocab.dummy)
			filtered_list.append(list(t))
			if len(val_data) < val_len:
				val_data.append(list(t))
			else:
				train_data.append(list(t))
				corpus.extend(sent)

		pbar.update(i)
	pbar.finish()

	print(len(filtered_list))

	logging.info("build corpus")
	vocab = Vocab(args.vocab, corpus)

	logging.info("encoding")
	train_data = vocab.encode_all_sents(train_data)
	val_data = vocab.encode_all_sents(val_data)
	train_data = np.array(train_data)
	val_data = np.array(val_data)

	print(train_data.shape)
	print(val_data.shape)
	logging.info("dump")
	pickle.dump(train_data, open(args.tr_data, "wb"))
	pickle.dump(val_data, open(args.val_data, "wb"))

	# split validation

	
	# val_data = filtered_list[:val_len]
	# train_data = filtered_list[val_len:]

	# build vocab for training data

	# corpus = [w for s in train_data for w in s]
	# print(corpus[:5])
	# print(len(corpus))













	# print(text_list[0])
	# corpus = " ".join(text_list)
	# logging.info("build counter")
	# cnt = Counter([stemmer.stem(w) for w in corpus.split()])
	# # cnt = Counter(corpus.split())
	# print(cnt.most_common(10))
	# filtered_list = []
	# sent_len = 0.0
	# sent_num = 0.0
	# dummy_num = 0.0

	# pbar = pb.ProgressBar(widgets=["[PREPROCESS] ", pb.DynamicMessage('sent_len'), " ", pb.DynamicMessage('dummy'), " ", pb.FileTransferSpeed(unit="sents"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(text_list)).start()
	# for i, sent in enumerate(text_list):
	# 	t = [vocab.start]
	# 	s_l = sent.split() + [vocab.end]

	# 	word_num = 0.0
		
	# 	for word in s_l:
	# 		if word == "" or word == " ":
	# 			continue
	# 		word = stemmer.stem(word)
			
	# 		if len(t) >= args.seq_size:
	# 			filtered_list.append(t)
	# 			# print(t)
	# 			t = [tt for tt in t]
	# 			t.pop(0)
	# 		if cnt[word] >= args.min_count:
	# 			t.append(word)
	# 			word_num += 1
	# 		else:
	# 			t.append(vocab.unknown)
	# 			word_num += 1
		
	# 	if len(t) >= 5:
				
	# 		while len(t) < args.seq_size:
	# 			t.append(vocab.dummy)
	# 			dummy_num += 1.0
	# 		sent_len = max(word_num, sent_len)

	# 		aa[word_num] = aa.get(word_num, 0) + 1
	# 		sent_num += 1.0
	# 		# print(t)
			
	# 		filtered_list.append(t)
	# 	pbar.update(i, sent_len=sent_len, dummy=sent_num)

	# pbar.finish()

	# print(len(filtered_list))


	
	# val_data = filtered_list[:val_len]
	# train_data = filtered_list[val_len:]
	# corpus = [w for s in train_data for w in s]
	# print(corpus[:5])
	# print(len(corpus))

	# vocab = Vocab(args.vocab, corpus)

	# train_data = vocab.encode_all_sents(train_data)
	# val_data = vocab.encode_all_sents(val_data)
	# train_data = np.array(train_data)
	# val_data = np.array(val_data)

	# print(train_data.shape)
	# print(val_data.shape)
	# pickle.dump(train_data, open(args.tr_data, "wb"))
	# pickle.dump(val_data, open(args.val_data, "wb"))

	# text_list = text_clear.split()
	# print(text_list[:20])
	# logging.info("build counter")
	# cnt = Counter(text_list)
	# filtered_list = []
	# pbar = pb.ProgressBar(widgets=["[PREPROCESS] ", pb.FileTransferSpeed(unit="words"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(text_list)).start()
	# for i, word in enumerate(text_list):
	# 	if cnt[word] >= args.min_count:
	# 		filtered_list.append(word)
	# 	pbar.update(i)
	# pbar.finish()

	# output_text = " ".join(filtered_list)
	# with open(args.tr_data_parsed, "w") as p:
	# 	p.write(output_text)

	# testing data

	test_data = []
	sent_len = 0
	with open(args.raw_te_data, "r") as f:
		reader = csv.DictReader(f)
		# id,question,a),b),c),d),e)
		for row in reader:
			question = row[question]
			for key in ["a)", "b)", "c)", "d)", "e)"]:
				# "_____"
				missing_word = row[key]
				sent = question.replace("_____", missing_word)
				sent_len = max(sent_len, len(sent.split()))
	print(sent_len)




if __name__ == "__main__":

	args = arg_parse()


	start_time = time.time()
	preprocessing(args)
	logging.info("training time: {}".format(time.time() - start_time))
