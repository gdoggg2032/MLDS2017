import tensorflow as tf 
from gan import Discriminator, Generator


class ConditionalWassersteinGAN(object):

	def __init__(self, z_dim, h_dim, learning_rate, scale, generator_output_layer):

		self.z_dim = z_dim
		self.h_dim = h_dim

		self.g_net = Generator(z_dim, h_dim, generator_output_layer)
		self.d_net = Discriminator(h_dim)

		self.training = tf.placeholder(tf.bool, [])

		self.with_text = tf.placeholder(tf.float32, [None])

		self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
		self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

		self.z = tf.placeholder(tf.float32, [None, self.z_dim])
		# true h
		self.h = tf.placeholder(tf.float32, [None, h_dim])
		# false h
		self.h_ = tf.placeholder(tf.float32, [None, h_dim])

		# false image
		self.x_ = self.g_net(self.z, self.h, self.training)

		# true image, true h
		self.d = self.d_net(self.x, self.h, self.training, reuse=False)

		# fake image, true h
		self.d_ = self.d_net(self.x_, self.h, self.training)

		# wrong image, true h
		self.d_w_ = self.d_net(self.x_w_, self.h, self.training)

		# true image, false h
		self.d_h_ = self.d_net(self.x, self.h_, self.training)

		self.g_loss = - tf.reduce_mean(self.d_) #+ tf.reduce_mean(tf.square(self.x - self.x_))
		self.d_loss = tf.reduce_mean(self.d) \
					- ( 1 * tf.reduce_mean(self.d_) + 1 * tf.reduce_mean(self.d_h_) + 1 * tf.reduce_mean(self.d_w_)) / (1 + 1 + 1)

		# penalty distribution for "improved wgan"

		epsilon = tf.random_uniform([], 0.0, 1.0)
		x_hat = epsilon * self.x + (1 - epsilon) * self.x_
		d_hat = self.d_net(x_hat, self.h, self.training)

		dx = tf.gradients(d_hat, x_hat)[0]
		dx_norm = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=[1,2,3]))
		
		ddx = scale * tf.reduce_mean(tf.square(dx_norm - 1.0))

		self.d_loss = -(self.d_loss - ddx)

		self.d_opt, self.g_opt = None, None
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.d_loss, var_list=self.d_net.vars)
			self.g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.g_loss, var_list=self.g_net.vars)


class ConditionalGAN(object):

	def __init__(self, z_dim, h_dim, learning_rate, scale, generator_output_layer):

		self.z_dim = z_dim
		self.h_dim = h_dim

		self.g_net = Generator(z_dim, h_dim, generator_output_layer)
		self.d_net = Discriminator(h_dim)

		self.training = tf.placeholder(tf.bool, [])

		self.with_text = tf.placeholder(tf.float32, [None])

		self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
		self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

		self.z = tf.placeholder(tf.float32, [None, self.z_dim])
		# true h
		self.h = tf.placeholder(tf.float32, [None, h_dim])
		# false h
		self.h_ = tf.placeholder(tf.float32, [None, h_dim])

		# false image
		self.x_ = self.g_net(self.z, self.h, self.training)

		# true image, true h
		self.d = self.d_net(self.x, self.h, self.training, reuse=False)

		# fake image, true h
		self.d_ = self.d_net(self.x_, self.h, self.training)

		# wrong image, true h
		self.d_w_ = self.d_net(self.x_w_, self.h, self.training)

		# true image, false h
		self.d_h_ = self.d_net(self.x, self.h_, self.training)

		# self.g_loss = - tf.reduce_mean(self.d_) #+ tf.reduce_mean(tf.square(self.x - self.x_))
		# self.d_loss = tf.reduce_mean(self.d) \
		# 			- ( 1 * tf.reduce_mean(self.d_) + 1 * tf.reduce_mean(self.d_h_) + 1 * tf.reduce_mean(self.d_w_)) / (1 + 1 + 1)

		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.ones_like(self.d_))) 

		self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
					+ (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.zeros_like(self.d_))) + \
					   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_w_, labels=tf.zeros_like(self.d_w_))) +\
					   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_h_, labels=tf.zeros_like(self.d_h_))) ) / 3 
		

		self.d_opt, self.g_opt = None, None
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.d_loss, var_list=self.d_net.vars)
			self.g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.g_loss, var_list=self.g_net.vars)


class ConditionalLSGAN(object):

	def __init__(self, z_dim, h_dim, learning_rate, scale, generator_output_layer):

		self.z_dim = z_dim
		self.h_dim = h_dim

		self.g_net = Generator(z_dim, h_dim, generator_output_layer)
		self.d_net = Discriminator(h_dim)

		self.training = tf.placeholder(tf.bool, [])

		self.with_text = tf.placeholder(tf.float32, [None])

		self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
		self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

		self.z = tf.placeholder(tf.float32, [None, self.z_dim])
		# true h
		self.h = tf.placeholder(tf.float32, [None, h_dim])
		# false h
		self.h_ = tf.placeholder(tf.float32, [None, h_dim])

		# false image
		self.x_ = self.g_net(self.z, self.h, self.training)

		# true image, true h
		self.d = self.d_net(self.x, self.h, self.training, reuse=False)

		# fake image, true h
		self.d_ = self.d_net(self.x_, self.h, self.training)

		# wrong image, true h
		self.d_w_ = self.d_net(self.x_w_, self.h, self.training)

		# true image, false h
		self.d_h_ = self.d_net(self.x, self.h_, self.training)

		# self.g_loss = - tf.reduce_mean(self.d_) #+ tf.reduce_mean(tf.square(self.x - self.x_))
		# self.d_loss = tf.reduce_mean(self.d) \
		# 			- ( 1 * tf.reduce_mean(self.d_) + 1 * tf.reduce_mean(self.d_h_) + 1 * tf.reduce_mean(self.d_w_)) / (1 + 1 + 1)

		# self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.ones_like(self.d_))) 

		# self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
		# 			+ (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.zeros_like(self.d_))) + \
		# 			   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_w_, labels=tf.zeros_like(self.d_w_))) +\
		# 			   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_h_, labels=tf.zeros_like(self.d_h_))) ) / 3 
		

		self.g_loss = 0.5 * tf.reduce_mean(tf.square(self.d_))

		self.d_loss = 0.5 * tf.reduce_mean(tf.square(self.d - 1)) + 0.5 * tf.reduce_mean(tf.square(self.d_ + 1)) \
					 + 0.5 * tf.reduce_mean(tf.square(self.d_w_ + 1)) + 0.5 * tf.reduce_mean(tf.square(self.d_h_ + 1))



		self.d_opt, self.g_opt = None, None
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.d_loss, var_list=self.d_net.vars)
			self.g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
				.minimize(self.g_loss, var_list=self.g_net.vars)



		
		







