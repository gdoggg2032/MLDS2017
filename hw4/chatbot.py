import tensorflow as tf 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logger.handlers[0] # stream handler
formatter = handler.formatter
handler = logging.FileHandler("./log")
handler.setFormatter(formatter)
logger.addHandler(handler)

import time
import os 
import progressbar as pb

from data_utils import DataManager

from collections import deque

# from nltk.translate.bleu_score import sentence_bleu
# from nltk.metrics import distance
# from reward import edit_distance

import numpy as np 

# from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat 


tf.flags.DEFINE_integer("mode", 2, "mode: 0=train & test, 1=test, 2=train")
tf.flags.DEFINE_integer("num_step", 12, "max timestep of sequence")
tf.flags.DEFINE_integer("min_count", 5, "min frequency of word in corpus")
tf.flags.DEFINE_integer("hidden_size", 256, "hidden dimension of LSTM Cell")
tf.flags.DEFINE_integer("vocab_dim", 100, "embedding dimension of vocab")
tf.flags.DEFINE_float("learning_rate", 1e-3, "initial learning rate")
tf.flags.DEFINE_float("clip_norm", 5.0, "gradient clip by norm")
tf.flags.DEFINE_float("discount_factor", 0.99, "reward discount factor (gamma)")
tf.flags.DEFINE_integer("replay_memory_size", 100000, "replay memory size")

tf.flags.DEFINE_integer("pretrain_actor_epochs", 100, "actor pretraining epochs")
tf.flags.DEFINE_integer("pretrain_critic_epochs", 100, "critic pretraining epochs")
tf.flags.DEFINE_integer("epochs", 1200, "total training epochs")
tf.flags.DEFINE_integer("epoch_history_size", 100, "record history for last N")
# tf.flags.DEFINE_integer("rollout_batch_size", 100, "batch size for Rollout")
tf.flags.DEFINE_integer("batch_size", 100, "batch size for replay memory training")


tf.flags.DEFINE_string("train_file_path", "./data/movie_lines_cleared.txt", "train data path")
tf.flags.DEFINE_string("test_file_path", "./sample_input.txt", "test_input_path")
tf.flags.DEFINE_string("test_output_path", "./sample_output_BEST.txt", "test_output_path")
tf.flags.DEFINE_string("vocab_path", "./vocab", "vocab path")
tf.flags.DEFINE_string("log", "./model_chatbot", "model directory path")
tf.flags.DEFINE_string("curve", "./assets/learning_curve_{}.png", "leanring curve for {} epochs")
tf.flags.DEFINE_string("model", "./models/model.ckpt", "model_name")
tf.flags.DEFINE_bool("restore", False, "whether resotre from FLAGS.model")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

logging.info(FLAGS.__flags)


def seq2seq_epoch():
	pass


reward_pool = {}
hit = 0


# def compute_reward_thread(arg):
# 	action, target, end_id = arg 

# 	action = list(action)


# 	acc_target = target[:target.index(end_id)]
# 	try:
# 		acc_action = action[:action.index(end_id)]
# 	except:
# 		acc_action = action 

# 	reward = []
# 	for i in range(len(acc_action)):
# 		try:
# 			r = sentence_bleu([acc_target], acc_action[:(i+1)])
# 		except:
# 			r = -0.01 * (i+1)
# 		reward.append(r)
	
# 	# append to max length
# 	if len(reward) > 0:
# 		reward = reward + [reward[-1]] * (len(action) - len(reward))
# 	else:
# 		reward = [0.] * (len(action) - len(reward))

# 	# reshape
# 	for i in range(len(reward)-1, 0, -1):
# 		reward[i] = reward[i] - reward[i - 1] 



# 	return reward

def compute_reward_thread(arg):
	action, target, end_id = arg 

	action = list(action)


	acc_target = target[:target.index(end_id)]
	try:
		acc_action = action[:action.index(end_id)]
	except:
		acc_action = action 

	reward = sentence_bleu([acc_target], acc_action)

	max_len = max(len(acc_target), len(acc_action))

	reward = [reward] * max_len + [0.] * (len(target) - max_len)



	return reward



	# reward = []
	# for a, t in zip(action, target):

	# 	if a == t and a == end_id:
	# 		reward.append(1)
	# 	elif a != t:
	# 		reward.append(-1)
	# 	elif a == t:
	# 		reward.append(1)

	# reward = [sum(reward)/len(reward)] * len(reward)

	# return reward

	# acc_target = target[:target.index(end_id)]
	# acc_action = action[:len(acc_target)]

	# reward = edit_distance(acc_action, acc_target)
	# print("a", reward)
	# # reward = reward + [reward[-1]] * (len(action) - len(reward))
	
	# for i in range(len(reward)-1, -1, -1):
	# 	if i == 0:
	# 		reward[i] = -(reward[i] - len(acc_target))
	# 	elif i > len(acc_action) - 1:
	# 		reward[i] = -1
	# 	else:
	# 		reward[i] = -(reward[i] - reward[i-1])
	# print("b", reward)
	# reward = reward + [0] * (len(action) - len(acc_action))
	# print("c", reward)
	# # acc_reward = [sum(reward[i:]) for i in range(len(reward))]
	
	# # total_abs_reward = abs(acc_reward[0]) 
	# # total_abs_reward = 1 if total_abs_reward == 0 else total_abs_reward
	# # acc_reward_normalized = [r/total_abs_reward for r in acc_reward]
	# # acc_reward_normalized = [1.0 for r in acc_reward]

	# return reward



# pool = ThreadPool(30) 

from collections import deque
reward_history = deque(maxlen=10000)

def policy_rollout(model, sess, data, end_id, random=True, greedy=False):
	"""Run a mini-batch data."""

	# global pool 

	global reward_history

	states = [d.state for d in data]
	targets = [d.target for d in data]
	dones = [True] * len(states)

	rewards = []
	greedy_rewards = []

	actions = None

	if random:

		actions = model.inference(sess, states, model='target')

		
		# for i, (a, t, e) in enumerate(zip(actions, targets, repeat(end_id, len(data)))):

		# 	reward = compute_reward_thread((a, t, e))
			
		# 	rewards.append(reward)
		# 	reward_history.append(reward)

		rewards = model.sentence_sim(sess, data, actions)



		# rewards = rewards + rewards2

		



	if greedy:

		greedy_actions = model.inference(sess, states, model='policy')

		

		for i, (g_a, t, e) in enumerate(zip(greedy_actions, targets, repeat(end_id, len(data)))):

			
			greedy_reward = compute_reward_thread((g_a, t, e))

			greedy_rewards.append(greedy_reward)


	return actions, rewards, greedy_rewards



	

	# if greedy:
	# 	greedy_rewards = []

	# 	rewards = []
	# 	for i, (a, t, e, g_a) in enumerate(zip(actions, targets, repeat(end_id, len(data)), greedy_actions)):

	# 		reward = compute_reward_thread((a, t, e))
	# 		greedy_reward = compute_reward_thread((g_a, t, e))

	# 		rewards.append(reward)

	# 		greedy_rewards.append(greedy_reward)
	# 		reward_history.append(reward)


	# 	return actions, rewards, greedy_rewards

	# else:

	# 	rewards = []
	# 	for i, (a, t, e) in enumerate(zip(actions, targets, repeat(end_id, len(data)))):

	# 		reward = compute_reward_thread((a, t, e))
			

	# 		rewards.append(reward)

	
	# 		reward_history.append(reward)

	# 	return actions, rewards


	# total_reward = np.mean([reward[0] for reward in rewards])



	# rewards = pool.map(compute_uni_reward_thread, zip(actions, targets, repeat(end_id)))
	# total_reward = np.mean(rewards)

	# for state, action, reward, target, done in zip(states, actions, rewards, targets, dones):
	# 	model.store_rollout(state, action, reward, target, done)
		# if sum(reward) < -1:
		# 	print(action, target, reward)
	# rewards = (rewards - np.mean(reward_history, axis=0)) 

	


def valid_epoch(dm, model, sess):

	global reward_history

	# random pick some sentence to show performace
	data = dm.draw_batch(5, mode='random')
	# data = dm.train_data
	states = [d.state for d in data]
	actions = model.inference(sess, states, model='policy')
	actions_target = model.inference(sess, states, model='target')
	targets = [d.target for d in data]

	estimated_values = model.get_predicted_taken_action_reward(sess, data, actions, model='policy')
	target_estimated_values = model.get_predicted_taken_action_reward(sess, data, actions_target, model='target')
	# decode to real words

	rewards = model.sentence_sim(sess, data, actions)
	target_rewards = model.sentence_sim(sess, data, actions_target)

	logging.info("show some results")

	for n, (s, a, a_t, t, e, e_t) in enumerate(zip(states, actions, actions_target, targets, estimated_values, target_estimated_values)):
		input_sent = " ".join([dm.vocab.decode(i) for i in s if i not in [dm.vocab.start_id, dm.vocab.end_id]])
		target_sent = " ".join([dm.vocab.decode(i) for i in t if i not in [dm.vocab.start_id, dm.vocab.end_id]])
		output_sent = " ".join([dm.vocab.decode(i) for i in a if i not in [dm.vocab.start_id, dm.vocab.end_id]])
		output_target_sent = " ".join([dm.vocab.decode(i) for i in a_t if i not in [dm.vocab.start_id, dm.vocab.end_id]])
		
		reward = rewards[n]#compute_reward_thread((a, t, dm.vocab.end_id)) 
		target_reward = target_rewards[n]#compute_reward_thread((a_t, t, dm.vocab.end_id)) 
		info_output = "sample {}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n".format(
			n, ("input", input_sent), ("target", target_sent), ("o", output_sent), ("o_t", output_target_sent), ("a", list(a)), ("a_t", list(a_t)), ("t", t), ("r", list(reward)), ("r_t", list(target_reward)), ("e", list(e)), ("e_t", list(e_t)))
		# if input_sent in ['hey sweet cheeks', 'where ve you been', 'you know chastity', 'well no']:
		logging.info(info_output)


def draw(log):

	import matplotlib as mpl
	mpl.use('Agg')

	import matplotlib.pyplot as plt
	epochs = len(log)

	max_bound = 1.0
	min_bound = 0.0

	state1 = True if epochs >= FLAGS.pretrain_actor_epochs else False
	state2 = True if epochs >= FLAGS.pretrain_actor_epochs + FLAGS.pretrain_critic_epochs else False
	state3 = True if state1 and state2 else False


	length = min(FLAGS.pretrain_actor_epochs, epochs)
	x = [i+1 for i in range(length)]
	y = log[0:length]
	plt.plot(x, y, label="pretrain-seq2seq")

	if epochs > FLAGS.pretrain_actor_epochs:

		length = min(FLAGS.pretrain_actor_epochs+FLAGS.pretrain_critic_epochs, epochs)
		x = [i+1 for i in range(FLAGS.pretrain_actor_epochs, length)]
		y = log[FLAGS.pretrain_actor_epochs:length]
		plt.plot(x, y, label="pretrain-critic")

		if epochs > FLAGS.pretrain_actor_epochs+FLAGS.pretrain_critic_epochs:

			x = [i+1 for i in range(FLAGS.pretrain_actor_epochs+FLAGS.pretrain_critic_epochs, epochs)]
			y = log[FLAGS.pretrain_actor_epochs+FLAGS.pretrain_critic_epochs:]
			plt.plot(x, y, label="actor-critic")

	plt.xlabel('Epochs')
	plt.ylabel('Reward(BLEU)')
	plt.axis([0, len(log)+10, 0, 1.0])
	plt.legend()

	plt.savefig(FLAGS.curve.format(epochs))
	plt.close()

def train():
	with tf.Graph().as_default():

		reward_log = []



		dm = DataManager(FLAGS.mode, FLAGS.num_step, FLAGS.train_file_path, vocab_path=FLAGS.vocab_path, min_count=FLAGS.min_count)

		from policy_gradient import PolicyGradientReinforce, PolicyGradientActorCritic
		model = PolicyGradientActorCritic(
				 FLAGS.num_step,
				 FLAGS.hidden_size,
				 dm.vocab.vocab_size,
				 FLAGS.vocab_dim,
				 dm.vocab.start_id,
				 dm.vocab.end_id,
				 FLAGS.learning_rate,
				 clip_norm=FLAGS.clip_norm,
				 discount_factor=FLAGS.discount_factor,
				 replay_memory_size=FLAGS.replay_memory_size)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=25)
		sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver, save_model_secs=10)

		print("tf.global_variables", [v.name for v in tf.global_variables()])
		print("tf.trainable_variables", [v.name for v in tf.trainable_variables()])
		print("tf.model_variables", [v.name for v in tf.model_variables()])

		

		with sv.managed_session(config=config) as sess:
		# with tf.Session(config=config) as sess:
		# for x in range(1):
			# sess = tf.Session()

			# if FLAGS.restore:
			# 	logging.info("load model from: {}".format(FLAGS.model))
			# 	saver.restore(sess, FLAGS.model)
			# else:
			# 	logging.info("initialize all variables")
			# 	sess.run(tf.global_variables_initializer())




			

			epoch_history = deque(maxlen=FLAGS.epoch_history_size)

			# first pretrain actor model
			for epoch in range(FLAGS.pretrain_actor_epochs):


				total_loss = 0.
				total_greedy_reward = 0.

				maxval = dm.total_batch_num(FLAGS.batch_size, mode='train')
				pbar = pb.ProgressBar(widgets=["[PretrainA, Epoch={}] ".format(epoch), pb.DynamicMessage('loss'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
				for i in range(maxval):
				
					data = dm.draw_batch(FLAGS.batch_size, mode='train')
					_, _, greedy_rewards = policy_rollout(model, sess, data, end_id=dm.vocab.end_id, random=False, greedy=True)
					loss = model.seq2seq_train(sess, data, train=True)

					total_loss += loss
					total_greedy_reward += np.mean([reward[0] for reward in greedy_rewards]) 

					pbar.update(i, loss=(total_loss / (i + 1)))
				pbar.finish()

				total_greedy_reward /= maxval
				reward_log.append(total_greedy_reward)
				epoch_history.append(total_greedy_reward)
				mean_reward = np.mean(epoch_history)

				valid_epoch(dm, model, sess)

				logging.info("Epoch {}".format(epoch))
				logging.info("A Loss {}".format(total_loss / maxval))
				logging.info("Greedy Reward for this epoch: {}".format(total_greedy_reward))
				logging.info("Average greedy reward for last {} epochs: {:.2f}".format(FLAGS.epoch_history_size, mean_reward))

				draw(reward_log)
				logging.info("save model to {}".format(saver.save(sess, FLAGS.model, global_step=epoch)))
				# logging.info("Reward for this epoch: {}".format(total_reward))
				# logging.info("Average reward for last {} epochs: {:.2f}".format(FLAGS.epoch_history_size, mean_reward))

			model.target_model_update(sess, 1.0)
			os.system("cp -r {} {}".format(FLAGS.log, "{}_A_{}-{}".format(FLAGS.log, FLAGS.pretrain_actor_epochs, "-".join(time.asctime().split()))))
			# input()

			# pretrain critic model
			for epoch in range(FLAGS.pretrain_actor_epochs, FLAGS.pretrain_actor_epochs + FLAGS.pretrain_critic_epochs):

				total_reward = 0.
				
				total_c_loss = 0.
				if epoch <= FLAGS.pretrain_actor_epochs + 1:
					total_greedy_reward = 0.

				maxval = dm.total_batch_num(FLAGS.batch_size, mode='train')
				pbar = pb.ProgressBar(widgets=["[RolloutC, Epoch={}] ".format(epoch), pb.DynamicMessage('c_loss'), " ", pb.FileTransferSpeed(unit="batchs"),  pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
				for i in range(maxval):

					data = dm.draw_batch(FLAGS.batch_size, mode='train')

					if epoch <= FLAGS.pretrain_actor_epochs + 1:
						actions, rewards, greedy_rewards = policy_rollout(model, sess, data, end_id=dm.vocab.end_id, greedy=True)
					else:
						actions, rewards, _ = policy_rollout(model, sess, data, end_id=dm.vocab.end_id)


					c_loss = model.update_model(sess, data, actions, rewards, both=False)
					model.target_model_update(sess, 1e-4)

					total_c_loss += c_loss

					total_reward += np.mean([reward[0] for reward in rewards]) 
					if epoch <= FLAGS.pretrain_actor_epochs + 1:
						total_greedy_reward += np.mean([reward[0] for reward in greedy_rewards])
					pbar.update(i, c_loss=(total_c_loss) / (i + 1))
				pbar.finish()

				total_reward /= maxval
				if epoch <= FLAGS.pretrain_actor_epochs + 1:
					total_greedy_reward /= maxval
					epoch_history.append(total_greedy_reward)
				reward_log.append(total_greedy_reward)
				mean_reward = np.mean(epoch_history)

				valid_epoch(dm, model, sess)

				logging.info("Epoch {}".format(epoch))
				logging.info("C Loss {}".format(total_c_loss / maxval))
				logging.info("Reward for this epoch: {}".format(total_reward))
				logging.info("Greedy Reward for this epoch: {}".format(total_greedy_reward))
				logging.info("Average greedy reward for last {} epochs: {:.2f}".format(FLAGS.epoch_history_size, mean_reward))

				draw(reward_log)
				logging.info("save model to {}".format(saver.save(sess, FLAGS.model, global_step=epoch)))


			# input()

			model.target_model_update(sess, 1.0)
			os.system("cp -r {} {}".format(FLAGS.log, "{}_A_{}_B_{}-{}".format(FLAGS.log, FLAGS.pretrain_actor_epochs, FLAGS.pretrain_critic_epochs, "-".join(time.asctime().split()))))

			# train both model

			for epoch in range(FLAGS.pretrain_actor_epochs + FLAGS.pretrain_critic_epochs, FLAGS.epochs):

				total_reward = 0.
				total_c_loss = 0.
				total_greedy_reward = 0.
				total_joint_score = 0.

				maxval = dm.total_batch_num(FLAGS.batch_size, mode='train')
				pbar = pb.ProgressBar(widgets=["[RolloutB, Epoch={}] ".format(epoch), pb.DynamicMessage('c_loss'), " ", pb.DynamicMessage('joint'), " ", pb.FileTransferSpeed(unit="batchs"),  pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
				for i in range(maxval):

					data = dm.draw_batch(FLAGS.batch_size, mode='train')

					actions, rewards, greedy_rewards = policy_rollout(model, sess, data, end_id=dm.vocab.end_id, greedy=True)

					c_loss, joint_score = model.update_model(sess, data, actions, rewards)
					model.target_model_update(sess, 1e-4)
					

					total_c_loss += c_loss
					total_joint_score += joint_score

					total_reward += np.mean([reward[0] for reward in rewards])
					total_greedy_reward += np.mean([reward[0] for reward in greedy_rewards]) 
					pbar.update(i, c_loss=(total_c_loss) / (i + 1), joint=(total_joint_score) / (i + 1))
				pbar.finish()

				total_reward /= maxval
				total_greedy_reward /= maxval
				reward_log.append(total_greedy_reward)
				epoch_history.append(total_greedy_reward)
				mean_reward = np.mean(epoch_history)

				valid_epoch(dm, model, sess)

				logging.info("Epoch {}".format(epoch))
				logging.info("C Loss {}".format(total_c_loss / maxval))
				logging.info("joint_score {}".format(total_joint_score / maxval))
				logging.info("Reward for this epoch: {}".format(total_reward))
				logging.info("Greedy Reward for this epoch: {}".format(total_greedy_reward))
				logging.info("Average greedy reward for last {} epochs: {:.2f}".format(FLAGS.epoch_history_size, mean_reward))

				draw(reward_log)
				logging.info("save model to {}".format(saver.save(sess, FLAGS.model, global_step=epoch)))
				
def test():
	
	with tf.Graph().as_default():


		dm = DataManager(FLAGS.mode, FLAGS.num_step, test_file_path=FLAGS.test_file_path, vocab_path=FLAGS.vocab_path, min_count=FLAGS.min_count)

		from policy_gradient import PolicyGradientReinforce, PolicyGradientActorCritic
		model = PolicyGradientActorCritic(
				 FLAGS.num_step,
				 FLAGS.hidden_size,
				 dm.vocab.vocab_size,
				 FLAGS.vocab_dim,
				 dm.vocab.start_id,
				 dm.vocab.end_id,
				 FLAGS.learning_rate,
				 clip_norm=FLAGS.clip_norm,
				 discount_factor=FLAGS.discount_factor,
				 replay_memory_size=FLAGS.replay_memory_size)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=25)
		sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver, save_model_secs=10)

		with sv.managed_session(config=config) as sess:

			p = open(FLAGS.test_output_path, "w")

			maxval = dm.total_batch_num(FLAGS.batch_size, mode='test')
			pbar = pb.ProgressBar(widgets=["[Test] ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
			for i in range(maxval):
			
				data = dm.draw_batch(FLAGS.batch_size, mode='test')

				states = [d.state for d in data]
				actions = model.inference(sess, states, model='policy')
				for a in actions:
					output_sent = " ".join([dm.vocab.decode(i) for i in a if i not in [dm.vocab.start_id, dm.vocab.end_id]])
					p.write(output_sent+'\n')


				pbar.update(i)
			pbar.finish()

			p.close()






if __name__ == "__main__":

	
	if FLAGS.mode % 2 == 0:

		if not os.path.exists("./assets"):
			os.makedirs("./assets")
		if not os.path.exists("./models"):
			os.makedirs("./models")
		s_time = time.time()
		train()
		logging.info("training time: {}".format(time.time() - s_time))


	if FLAGS.mode // 2 == 0:
		s_time = time.time()
		test()
		logging.info("testing time: {}".format(time.time() - s_time))

