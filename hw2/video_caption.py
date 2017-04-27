import tensorflow as tf 
import logging
logging.basicConfig(level=logging.INFO)
import time
from data_utils import VideoCaptionDataManager
from mixer import MIXER
import numpy as np 
from replay import Sample, ReplayMemory

import progressbar as pb


from nltk.translate.bleu_score import sentence_bleu

import difflib




tf.flags.DEFINE_integer("mode", 2, "mode: 0=train & test, 1=test, 2=train")
tf.flags.DEFINE_integer("max_step", 15, "max output label length")
tf.flags.DEFINE_integer("init_epochs", 4, "initial iteration number for fully XENT training")
tf.flags.DEFINE_integer("mix_epochs", 2, "max iteration number")
tf.flags.DEFINE_integer("batch_size", 100, "replay batch size for training")
tf.flags.DEFINE_integer("hidden_size", 256, "LSTM hidden size")
tf.flags.DEFINE_integer("dim", 300, "embedding dimension")
tf.flags.DEFINE_integer("encoder_input_len", 80, "encoder input sequence length")
tf.flags.DEFINE_integer("encoder_input_size", 4096, "encoder input dimension")
tf.flags.DEFINE_integer("target_model_update_freq", 10, "target update after several epochs")
tf.flags.DEFINE_integer("min_count", 3, "min word frequency for building vocab")
tf.flags.DEFINE_integer("replay_capacity", 400000, "replay memory size")
tf.flags.DEFINE_integer("border_step", 1, "xent_border shift step")
tf.flags.DEFINE_bool("prioritized_replay", False, "replay prioritized draw batch")
tf.flags.DEFINE_float("learning_rate", 1e-2, "DQN RMSProp learning rate")
tf.flags.DEFINE_float("gamma", 0.99, "reward decay rate")


tf.flags.DEFINE_string("train_label", "./MLDS_hw2_data/training_label.json", "training label json file")
tf.flags.DEFINE_string("train_feat", "./MLDS_hw2_data/training_data/feat", "training feature directory")
tf.flags.DEFINE_string("valid_label", "./MLDS_hw2_data/testing_public_label.json", "validation label json file")
tf.flags.DEFINE_string("valid_feat", "./MLDS_hw2_data/testing_data/feat", "validation feature directory")
tf.flags.DEFINE_string("test_id", "./MLDS_hw2_data/testing_id.txt", "testing label json file")
tf.flags.DEFINE_string("test_feat", "./MLDS_hw2_data/testing_data/feat", "testing feature directory")
tf.flags.DEFINE_string("vocab", "./vocab", "Model vocab path")

tf.flags.DEFINE_string("rein_target_score_name", "bleu-4", "reinforce optimization score name, bleu-1, bleu-4, f1 or lcs")

tf.flags.DEFINE_string("log", "./log", "Model log directory")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def add_batch_samples(replay_memory, 
	encoder_inputs, 
	input_c_states, input_h_states,
	output_c_states, output_h_states,
	actions, rewards, terminals, prev_terminals):
	for e, ic, ih, oc, oh, a, r, t, pt in zip(encoder_inputs, 
	input_c_states, input_h_states,
	output_c_states, output_h_states,
	actions, rewards, terminals, prev_terminals):
		if pt != 1:
			sample = Sample(e, ic, ih, oc, oh, a, r, t)
			replay_memory.add_sample(sample)
	

def bleu_4(preds, labels, end_id):

	scores = np.zeros(len(preds))
	for i, (p, s) in enumerate(zip(preds, labels)):
		try:
			p = p[:p.index(end_id)]
		except:
			p = p
		try:
			s = s[:s.index(end_id)]
		except:
			s = s
		scores[i] = sentence_bleu([s], p)

	return scores

def lcs(preds, labels, end_id):
	scores = np.zeros(len(preds))
	for i, (p, s) in enumerate(zip(preds, labels)):
		try:
			p = p[:p.index(end_id)]
		except:
			p = p
		try:
			s = s[:s.index(end_id)]
		except:
			s = s
		sm = difflib.SequenceMatcher()
		sm.set_seqs(p, s) 
		scores[i] = sum([x.size for x in sm.get_matching_blocks()])
	return scores

def run_xent_epoch(epoch, data_manager, model, replay_memory, sess, xent_border, train=True):

	total_loss = 0.0
	maxval = data_manager.total_batch['train']
	pbar = pb.ProgressBar(widgets=["[TRAIN XENT {}, B={}] ".format(epoch, xent_border),  pb.DynamicMessage('loss'), " ", pb.DynamicMessage('memory'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()

	for i in range(data_manager.total_batch['train']):

		batch = data_manager.draw_batch(FLAGS.batch_size)

	

		loss, c_states, h_states, preds = model.train_xent(sess, batch, xent_border, train)
		total_loss += loss

		# preds: all end_id after xent_border

		# compute score
		labels = np.array([b.labels for b in batch])
		# use step answer or full answer?
		# seq_labels = np.array([b.labels[:xent_border]+[data_manager.end_id]*(FLAGS.max_step - xent_border) for b in batch])
		# seq_labels = labels

		if FLAGS.rein_target_score_name == "bleu-1":
			prev_scores = model.bleu(sess, preds, labels)
		elif FLAGS.rein_target_score_name == "bleu-4":
			prev_scores = bleu_4(preds, labels, data_manager.end_id)
		elif FLAGS.rein_target_score_name == "f1":
			prev_scores = model.f1(sess, preds, labels)
		elif FLAGS.rein_target_score_name == 'lcs':
			prev_scores = lcs(preds, labels, data_manager.end_id)

		encoder_inputs = [b.frames for b in batch]

		

		if xent_border < FLAGS.max_step:
			terminals = np.zeros(len(batch))
			prev_terminals = np.zeros(len(batch))
			for t in range(xent_border, FLAGS.max_step):
				output_c_states, output_h_states, pred = model.inference(sess, encoder_inputs, c_states, h_states)
				preds[:, t-1] = pred
		
				if t == FLAGS.max_step -1:
					prev_terminals = terminals
					terminals = np.ones(len(batch), dtype='int')
				else:
					prev_terminals = terminals
					terminals = np.logical_or(pred == data_manager.end_id, terminals).astype('int')

				if FLAGS.rein_target_score_name == "bleu-1":
					scores = model.bleu(sess, preds, labels)
				elif FLAGS.rein_target_score_name == "bleu-4":
					scores = bleu_4(preds, labels, data_manager.end_id)
				elif FLAGS.rein_target_score_name == "f1":
					scores = model.f1(sess, preds, labels)
				elif FLAGS.rein_target_score_name == 'lcs':
					scores = lcs(preds, labels, data_manager.end_id)

				# only when terminal, get rewards
				# rewards = scores * teiminals
				# accumulative rewards
				rewards = scores - prev_scores

				add_batch_samples(replay_memory, encoder_inputs, c_states, h_states, output_c_states, output_h_states, pred, rewards, terminals, prev_terminals)
				c_states, h_states = output_c_states, output_h_states
				prev_scores = scores

		pbar.update(i, loss=total_loss/(i+1), memory=replay_memory.num_samples())
	pbar.finish()

	replay_memory.truncate_list_if_necessary()
						
				


	return total_loss / data_manager.total_batch['train']

def run_rein_epoch(epoch, model, sess, replay_memory, xent_border):

	total_loss = 0.0

	total_batch = int(np.ceil(replay_memory.num_samples() / FLAGS.batch_size))
	
	maxval = total_batch
	pbar = pb.ProgressBar(widgets=["[TRAIN REIN {}, B={}] ".format(epoch, xent_border),  pb.DynamicMessage('loss'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()


	for i in range(total_batch):
		batch = replay_memory.draw_batch(FLAGS.batch_size)
		loss = model.train_rein(sess, batch)
		total_loss += loss

		if i % FLAGS.target_model_update_freq == 0:
				model.target_model_update(sess)
				info_output = "REIN EPOCH {} target model update".format(epoch)
				logging.info(info_output)

		pbar.update(i, loss=total_loss/(i+1))
	pbar.finish()
					
	return total_loss / total_batch

def evaluation(model, sess, data_manager, xent_border, score_name, print_sample=True, eval_type='valid'):

	total_score = 0.0
	total_loss = 0.0

	length = len(data_manager.valid_data) if eval_type == 'valid' else len(data_manager.train_data)

	for i in range(data_manager.total_batch[eval_type]):
		batch = data_manager.draw_batch(FLAGS.batch_size, mode=eval_type)
		loss, preds = model.eval(sess, batch, xent_border)
		labels = [b.labels for b in batch]

		if score_name == "bleu-1":
			scores = model.bleu(sess, preds, labels)
		elif score_name == "bleu-4":
			scores = bleu_4(preds, labels, data_manager.end_id)
		elif score_name == "f1":
			scores = model.f1(sess, preds, labels)
		elif score_name == 'lcs':
			scores = lcs(preds, labels, data_manager.end_id)
		# scores = model.bleu(sess, preds, labels)
		# scores = model.f1(sess, preds, labels)
		total_score += np.sum(scores)
		total_loss += loss * len(batch)


		if print_sample:
			# random choose 1 video output
			index = np.random.choice(len(batch), 1)[0]
			sample = batch[i]
			pred = preds[i]
			truth = sample.labels
			video_id = sample.video_id
			pred_sent = data_manager.index_to_string(pred)
			true_sent = data_manager.index_to_string(truth)
			score = scores[i]
			info_output = "video_id: {}\nTRUTH: {}\nPREDICT: {}\n{}: {}".format(video_id, true_sent, pred_sent, FLAGS.rein_target_score_name, score)
			logging.info(info_output)

	return total_score / length, total_loss / length



def train():
	with tf.Graph().as_default():

		data_manager = VideoCaptionDataManager(FLAGS)

		model = MIXER(FLAGS, data_manager.num_actions, data_manager.start_id, data_manager.end_id)

		replay_memory = ReplayMemory(FLAGS.replay_capacity, FLAGS.prioritized_replay)

		with model.sv.managed_session(config=model.config) as sess:

			for border in range(FLAGS.max_step, -1, -FLAGS.border_step):

				if border == FLAGS.max_step:
					# xent part
					for epoch in range(FLAGS.init_epochs):
						loss = run_xent_epoch(epoch, data_manager, model, replay_memory, sess, border)
						info_output = "XENT border: {}, EPOCH {} cross entropy xent_loss: {}".format(border, epoch, loss)
						logging.info(info_output)



						model.initial_rein_weight_update(sess)

						

						score, loss = evaluation(model, sess, data_manager, border, FLAGS.rein_target_score_name, print_sample=True)
						info_output = "VALID {}: {}, loss: {}".format(FLAGS.rein_target_score_name, score, loss)
						logging.info(info_output)
						score, loss = evaluation(model, sess, data_manager, border, FLAGS.rein_target_score_name, print_sample=False, eval_type='train')
						info_output = "TRAIN {}: {}, loss: {}".format(FLAGS.rein_target_score_name, score, loss)
						logging.info(info_output)
						score, loss = evaluation(model, sess, data_manager, border, "bleu-1", print_sample=False)
						info_output = "VALID {}: {}, loss: {}".format("bleu-1", score, loss)
						logging.info(info_output)


				if border < FLAGS.max_step:

					for epoch in range(FLAGS.mix_epochs):

						replay_memory.samples = []
						
						# first train cross entropy part and add some samples in replay memory
						loss = run_xent_epoch(epoch, data_manager, model, replay_memory, sess, border, train=False)
						info_output = "XENT border: {}, EPOCH {} cross entropy xent_loss: {}".format(border, epoch, loss)
						logging.info(info_output)
						# score, loss = evaluation(model, sess, data_manager, border, FLAGS.rein_target_score_name, print_sample=True)
						# info_output = "VALID {}: {}, loss: {}".format(FLAGS.rein_target_score_name, score, loss)
						# logging.info(info_output)
						# score, loss = evaluation(model, sess, data_manager, border, "bleu-1", print_sample=False)
						# info_output = "VALID {}: {}, loss: {}".format("bleu-1", score, loss)
						# logging.info(info_output)

						# train rein part
						loss = run_rein_epoch(epoch, model, sess, replay_memory, border)
						info_output = "REIN border: {}, EPOCH {} l2 loss: {}".format(border, epoch, loss)
						logging.info(info_output)

						# if epoch % FLAGS.target_model_update_freq == 0:
						# 	model.target_model_update(sess)
						# 	info_output = "REIN EPOCH {} target model update".format(epoch)
						# 	logging.info(info_output)
						
						score, loss = evaluation(model, sess, data_manager, border, FLAGS.rein_target_score_name, print_sample=True)
						info_output = "VALID {}: {}, loss: {}".format(FLAGS.rein_target_score_name, score, loss)
						logging.info(info_output)
						score, loss = evaluation(model, sess, data_manager, border, FLAGS.rein_target_score_name, print_sample=False, eval_type='train')
						info_output = "TRAIN {}: {}, loss: {}".format(FLAGS.rein_target_score_name, score, loss)
						logging.info(info_output)
						score, loss = evaluation(model, sess, data_manager, border, "bleu-1", print_sample=False)
						info_output = "VALID {}: {}, loss: {}".format("bleu-1", score, loss)
						logging.info(info_output)



def test():

	pass

if __name__ == "__main__":

	
	if FLAGS.mode % 2 == 0:
		s_time = time.time()
		train()
		logging.info("training time: {}".format(time.time() - s_time))


	if FLAGS.mode // 2 == 0:
		s_time = time.time()
		test()
		logging.info("testing time: {}".format(time.time() - s_time))


