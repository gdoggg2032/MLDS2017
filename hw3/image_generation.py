import tensorflow as tf 
import time
import os
import logging
logging.basicConfig(level=logging.INFO)

from data_utils import DataManager


from scipy.misc import imsave
import skimage.transform

import skimage.color

import progressbar as pb

import numpy as np

tf.flags.DEFINE_integer("mode", 0, "mode: 0=train & test, 1=test, 2=train")
tf.flags.DEFINE_integer("epochs", 60, "discriminator update epochs")
tf.flags.DEFINE_integer("d_epochs", 1, "discriminator update epochs")
tf.flags.DEFINE_integer("g_epochs", 1, "generator update epochs")
tf.flags.DEFINE_integer("batch_size", 100, "replay batch size for training")
tf.flags.DEFINE_integer("z_dim", 100, "z dimension")

tf.flags.DEFINE_integer('seed', None, "random seed")

tf.flags.DEFINE_float("scale", 10.0, "scalar for improved wgan penalty loss combination")
tf.flags.DEFINE_float("learning_rate", 2e-4, "learning rate of Generator and Discriminator")

tf.flags.DEFINE_string("z_type", "normal", "z sampler distribution type: [truncnorm, uniform, normal]")

tf.flags.DEFINE_string("tag_file", "./data/tags_clean.csv", "training tags path")
tf.flags.DEFINE_string("img_dir", "./data/faces/", "training images path")
tf.flags.DEFINE_string("test_text", "./data/sample_testing_text.txt", "testing text path")
tf.flags.DEFINE_string("vocab", "./vocab", "Model vocab path")
tf.flags.DEFINE_string("log", "./log", "Model log directory")

tf.flags.DEFINE_string("test_img_dir", "./samples/", "Model test sample images directory")
tf.flags.DEFINE_string("valid_img_dir", "./valid_samples/", "Model validation sample images directory")

tf.flags.DEFINE_string("model_type", "dcwgan", "model type: [dcgan, dcwgan, dclsgan]")

tf.flags.DEFINE_string("generator_output_layer", 'tanh', 'generator_output_layer: [tanh, sigmoid]')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.seed:
	np.random.seed(seed=FLAGS.seed)


def run_train_epoch(epoch, sess, dm, model):

	total_batch_num = dm.total_batch_num(FLAGS.batch_size)
	maxval = total_batch_num
	pbar = pb.ProgressBar(widgets=["[TRAIN {}] ".format(epoch), pb.DynamicMessage('d_loss'), " ", pb.DynamicMessage('g_loss'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
	total_d_loss = 0
	d_count = 0
	total_g_loss = 0
	g_count = 0
	for i in range(total_batch_num):

		data, bz, bh_, bx_w_ = dm.draw_batch(FLAGS.batch_size, FLAGS.z_dim, mode='train')
		bx = [d.img for d in data]
		bh = [d.tags for d in data]
		# with soft label
		# noise_range = 0.1
		# noise = np.random.uniform(-noise_range, noise_range)
		# bh = [d.tags * (1 - noise_range) + noise_range + noise for d in data]

		# bh_ = [h * (1 - noise_range) + noise_range + noise for h in bh_]

		bwith_text = [d.with_text for d in data]
		
		# if d_count == 0 or (total_d_loss / d_count) == 0 or total_g_loss / g_count / (total_d_loss / d_count) < 1.5:
		if i % 1 == 0:
			for d_i in range(FLAGS.d_epochs):
				_, d_loss = sess.run([model.d_opt, model.d_loss], 
					feed_dict={
						model.x:bx,
						model.z:bz,
						model.h:bh,
						model.h_:bh_,
						model.x_w_:bx_w_,
						model.training:True,
						model.with_text:bwith_text
					}
				)
				total_d_loss += d_loss
				d_count += 1


		for g_i in range(FLAGS.g_epochs):

			_, g_loss = sess.run([model.g_opt, model.g_loss], 
				feed_dict={
					model.x:bx,
					model.z:bz,
					model.h:bh,
					model.h_:bh_,
					model.x_w_:bx_w_,
					model.training:True,
					model.with_text:bwith_text
				}
			)
			total_g_loss += g_loss
			g_count += 1

		pbar.update(i, d_loss=total_d_loss / d_count, g_loss=total_g_loss / g_count)
	pbar.finish()

	return total_d_loss / d_count, total_g_loss / g_count


def run_valid(epoch, sess, dm, model):

	if not os.path.exists(FLAGS.valid_img_dir):
		os.makedirs(FLAGS.valid_img_dir)

	info_output = "[VALID]EPOCH {} save images in {}".format(epoch, FLAGS.valid_img_dir)
	logging.info(info_output)

	# random pick 5 images
	data, bz = dm.draw_batch(10, FLAGS.z_dim, mode='random')
	bh = [d.tags for d in data]
	images = sess.run(model.x_, 
		feed_dict={
			model.z:bz,
			model.h:bh,
			model.training:False
		}
	)

	# save images
	for image, d in zip(images, data):

		# convert image from -1, 1 to 0, 1
		if FLAGS.generator_output_layer == 'tanh':
			image = (image + 1.0) / 2.0
		elif FLAGS.generator_output_layer == 'sigmoid':
			image = image

		# resize image
		# img_resized = skimage.transform.resize(image, (64, 64), mode='constant')
		# do not resize 
		img_resized = image

		tag_text = d.tag_text
		img_id = d.img_id
		img_filename = "{}_{}_{}.jpg".format(epoch, img_id, tag_text)
		logging.info(img_filename)
		# imsave(os.path.join(FLAGS.valid_img_dir, img_filename), image)
		imsave(os.path.join(FLAGS.valid_img_dir, img_filename), img_resized)

	# generate tracked images for gif
	data, bz = dm.draw_batch(10, FLAGS.z_dim, mode='gif')
	bh = [d.tags for d in data]
	images = sess.run(model.x_, 
		feed_dict={
			model.z:bz,
			model.h:bh,
			model.training:False
		}
	)

	# save images
	for image, d in zip(images, data):

		# convert image from -1, 1 to 0, 1
		if FLAGS.generator_output_layer == 'tanh':
			image = (image + 1.0) / 2.0
		elif FLAGS.generator_output_layer == 'sigmoid':
			image = image

		# resize image
		# img_resized = skimage.transform.resize(image, (64, 64), mode='constant')
		# do not resize 
		img_resized = image
		
		tag_text = d.tag_text
		img_id = d.img_id
		img_filename = "GIF_{}_{}_{}.jpg".format(img_id, tag_text, epoch)
		logging.info(img_filename)
		# imsave(os.path.join(FLAGS.valid_img_dir, img_filename), image)
		imsave(os.path.join(FLAGS.valid_img_dir, img_filename), img_resized)


def train():

	dm = DataManager(FLAGS.mode, 
		FLAGS.tag_file, FLAGS.img_dir, FLAGS.test_text, FLAGS.vocab, FLAGS.z_dim, FLAGS.z_type, FLAGS.generator_output_layer)

	with tf.Graph().as_default():

		if FLAGS.model_type == 'dcwgan':
			from cond_wgan import ConditionalWassersteinGAN

			model = ConditionalWassersteinGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

		elif FLAGS.model_type == 'dcgan':
			from cond_wgan import ConditionalGAN

			model = ConditionalGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)
		
		elif FLAGS.model_type == 'dclsgan':
			from cond_wgan import ConditionalLSGAN

			model = ConditionalLSGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)
		

		
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		
		saver = tf.train.Saver(max_to_keep=25)
		
		sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver)
		
		with sv.managed_session(config=config) as sess:

			for epoch in range(FLAGS.epochs):
				
				d_loss, g_loss = run_train_epoch(epoch, sess, dm, model)
				info_output = "EPOCH {} d_loss: {}, g_loss: {}".format(epoch, d_loss, g_loss)
				logging.info(info_output)

				# random generate some image for evaluation
				run_valid(epoch, sess, dm, model)


def run_test_epoch(sess, dm, model):

	if not os.path.exists(FLAGS.test_img_dir):
		os.makedirs(FLAGS.test_img_dir)

	info_output = "[TEST]save images in {}".format(FLAGS.test_img_dir)
	logging.info(info_output)

	total_batch_num = dm.total_batch_num(FLAGS.batch_size, mode='test')
	maxval = total_batch_num
	pbar = pb.ProgressBar(widgets=["[TEST]", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=maxval).start()
	

	for i in range(total_batch_num):

		data, bz = dm.draw_batch(FLAGS.batch_size, FLAGS.z_dim, mode='test')
		bh = [d.tags for d in data]
		

		images = sess.run(model.x_, 
			feed_dict={
				model.z:bz,
				model.h:bh,
				model.training:False
			}
		)

		# save images
		for i, (image, d) in enumerate(zip(images, data)):

			# convert image from -1, 1 to 0, 1

			if FLAGS.generator_output_layer == 'tanh':
				image = (image + 1.0) / 2.0
			elif FLAGS.generator_output_layer == 'sigmoid':
				image = image
			

			# resize image
			# img_resized = skimage.transform.resize(image, (64, 64), mode='constant')
			# do not resize 
			img_resized = image

			tag_text = d.tag_text
			img_id = d.img_id
			img_filename = "sample_{}.jpg".format(img_id)
			logging.info("{}/{}: {}".format(i + 1, len(dm.test_data), img_filename))
			# imsave(os.path.join(FLAGS.valid_img_dir, img_filename), image)
			imsave(os.path.join(FLAGS.test_img_dir, img_filename), img_resized)
		pbar.update(i)
	pbar.finish()

def test():

	dm = DataManager(FLAGS.mode, 
		FLAGS.tag_file, FLAGS.img_dir, FLAGS.test_text, FLAGS.vocab, FLAGS.z_dim, FLAGS.z_type, FLAGS.generator_output_layer)

	with tf.Graph().as_default():

		if FLAGS.model_type == 'dcwgan':
			from cond_wgan import ConditionalWassersteinGAN

			model = ConditionalWassersteinGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

		elif FLAGS.model_type == 'dcgan':
			from cond_wgan import ConditionalGAN

			model = ConditionalGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

		elif FLAGS.model_type == 'dclsgan':
			from cond_wgan import ConditionalLSGAN

			model = ConditionalLSGAN(
				FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)
		
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		
		saver = tf.train.Saver(max_to_keep=25)

		def load_pretrain(sess):
			saver.restore(sess, os.path.join(FLAGS.log, "checkpoint"))

		sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver, init_fn=load_pretrain)
		
		with sv.managed_session(config=config) as sess:

			run_test_epoch(sess, dm, model)


if __name__ == "__main__":

	if FLAGS.mode % 2 == 0:
		s_time = time.time()
		train()
		logging.info("training time: {}".format(time.time() - s_time))


	if FLAGS.mode // 2 == 0:
		s_time = time.time()
		test()
		logging.info("testing time: {}".format(time.time() - s_time))
