import tensorflow as tf 
from tensorflow import layers



def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Discriminator(object):

	def __init__(self, h_dim):
		self.x_dim = [None, 64, 64, 3]
		self.h_dim = h_dim
		self.name = 'dcgan/d_net'

	def __call__(self, x, h, training, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
			# x = placeholder

			

			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 64, 64, 1])

			# x_h = tf.concat([x, r_h], axis=3)

			conv1 = layers.conv2d(
				x, 64, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)
			# batch_norm
			conv1_batch_norm = layers.batch_normalization(
				conv1, training=training
			)
			conv1_a = leaky_relu(conv1_batch_norm)

			# conv1_a: (None, 32, 32, 64)

			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

			# conv1_a_h = tf.concat([conv1_a, r_h], axis=3)


			conv2 = layers.conv2d(
				conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv2_batch_norm = layers.batch_normalization(
				conv2, training=training
			)
			conv2_a = leaky_relu(conv2_batch_norm)

			# conv2_a: (None, 16, 16, 128)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

			conv2_a_h = tf.concat([conv2_a, r_h], axis=3)

			conv3 = layers.conv2d(
				conv2_a_h, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv3_batch_norm = layers.batch_normalization(
				conv3, training=training
			)
			conv3_a = leaky_relu(conv3_batch_norm)

			# conv3_a: (None, 8, 8, 64 * 4)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

			conv3_a_h = tf.concat([conv3_a, r_h], axis=3)

			conv4 = layers.conv2d(
				conv3_a_h, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv4_batch_norm = layers.batch_normalization(
				conv4, training=training
			)


			conv4_1 = layers.conv2d(
				conv4_batch_norm, 64 * 2, kernel_size=[1, 1], strides=[1, 1],
				padding='valid',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv4_1_batch_norm = layers.batch_normalization(
				conv4_1, training=training
			)

			conv4_1_a = leaky_relu(conv4_1_batch_norm)

			conv4_2 = layers.conv2d(
				conv4_1_a, 64 * 2, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv4_2_batch_norm = layers.batch_normalization(
				conv4_2, training=training
			)

			conv4_2_a = leaky_relu(conv4_2_batch_norm)

			conv4_3 = layers.conv2d(
				conv4_2_a, 64 * 8, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv4_3_batch_norm = layers.batch_normalization(
				conv4_3, training=training
			)

	


			conv4_a = leaky_relu(conv4_batch_norm + conv4_3_batch_norm)

			# conv4_a: (None, 4, 4, 64 * 8)






			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

			concat_h = tf.concat([conv4_a, r_h], axis=3)

			# concat_h: (None, 4, 4, 64 * 8 + h_dim)

			conv5 = layers.conv2d(
				concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv5_batch_norm = layers.batch_normalization(
				conv5, training=training
			)
			conv5_a = leaky_relu(conv5_batch_norm)

			# conv5_a: (None, 4, 4, 64 * 8)

			# conv6 = layers.conv2d(
			# 	conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
			# 	padding='valid',
			# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			# 	activation=None
			# )

			# return conv6


			conv6 = layers.conv2d(
				conv5_a, self.h_dim, kernel_size=[4, 4], strides=[1, 1],
				padding='valid',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# conv6: (None, 1, 1, 225)


			conv6_sq = tf.squeeze(conv6, [1, 2])

			# conv6_sq: (None, 225)

			# h_h = tf.reshape(tf.matmul(h[:, :15, None], h[:, None, 15:]), [-1, 15 * 15])


			fetch = tf.reduce_sum(tf.multiply(conv6_sq, h), axis=1)

			# # fetch: (None)

			return fetch





	@property 
	def vars(self):

		return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):

	def __init__(self, z_dim, h_dim):
		self.z_dim = z_dim
		self.h_dim = h_dim
		self.x_dim = [None, 64, 64, 3]
		self.name = 'dcgan/g_net'

	def __call__(self, z, h, training):
		
		with tf.variable_scope(self.name) as vs:

			# concat z, h
			z_h = tf.concat([z, h], axis=1)

			# fc
			fc1 = layers.dense(
				z_h, 64 * 8 * 4 * 4,
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			fc1_batch_norm = layers.batch_normalization(
				fc1, training=training
			)

			fc1_conv = tf.reshape(fc1_batch_norm, [-1, 4, 4, 64 * 8])

			fc1_1conv = layers.conv2d(
				fc1_conv, 64 * 2, kernel_size=[1, 1], strides=[1, 1],
				padding='valid',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			fc1_1conv_batch_norm = layers.batch_normalization(
				fc1_1conv, training=training
			)

			fc1_1conv_a = tf.nn.relu(fc1_1conv_batch_norm)

			fc1_2conv = layers.conv2d(
				fc1_1conv_a, 64 * 2, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			fc1_2conv_batch_norm = layers.batch_normalization(
				fc1_2conv, training=training
			)

			fc1_2conv_a = tf.nn.relu(fc1_2conv_batch_norm)

			fc1_3conv = layers.conv2d(
				fc1_2conv_a, 64 * 8, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			fc1_3conv_batch_norm = layers.batch_normalization(
				fc1_3conv, training=training
			)


			fc1_a = tf.nn.relu(fc1_conv + fc1_3conv_batch_norm)

			# fc1_a: (None, 64 * 8 * 4 * 4)

			# conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])
			conv = fc1_a

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

			conv_h = tf.concat([conv, r_h], axis=3)

			conv1 = layers.conv2d_transpose(
				conv_h, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv1_batch_norm = layers.batch_normalization(
				conv1, training=training
			)



			conv1_1conv = layers.conv2d(
				conv1_batch_norm, 64, kernel_size=[1, 1], strides=[1, 1],
				padding='valid',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv1_1conv_batch_norm = layers.batch_normalization(
				conv1_1conv, training=training
			)

			conv1_1conv_a = tf.nn.relu(conv1_1conv_batch_norm)

			conv1_2conv = layers.conv2d(
				conv1_1conv_a, 64, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv1_2conv_batch_norm = layers.batch_normalization(
				conv1_2conv, training=training
			)

			conv1_2conv_a = tf.nn.relu(conv1_2conv_batch_norm)

			conv1_3conv = layers.conv2d(
				conv1_2conv_a, 64 * 4, kernel_size=[3, 3], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			conv1_3conv_batch_norm = layers.batch_normalization(
				conv1_3conv, training=training
			)

			conv1_a = tf.nn.relu(conv1_batch_norm + conv1_3conv_batch_norm)

			# conv1_a: (None, 8, 8, 64 * 4)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

			conv1_a_h = tf.concat([conv1_a, r_h], axis=3)

			conv2 = layers.conv2d_transpose(
				conv1_a_h, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv2_batch_norm = layers.batch_normalization(
				conv2, training=training
			)
			conv2_a = tf.nn.relu(conv2_batch_norm)

			# conv2_a: (None, 16, 16, 64 * 2)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

			conv2_a_h = tf.concat([conv2_a, r_h], axis=3)


			conv3 = layers.conv2d_transpose(
				conv2_a_h, 64, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv3_batch_norm = layers.batch_normalization(
				conv3, training=training
			)
			conv3_a = tf.nn.relu(conv3_batch_norm)

			# conv3: (None, 32, 32, 64)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

			conv3_a_h = tf.concat([conv3_a, r_h], axis=3)			

			conv4 = layers.conv2d_transpose(
				conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv4_batch_norm = layers.batch_normalization(
				conv4, training=training
			)
			# conv4_a = tf.nn.relu(conv4_batch_norm)

			# conv4: (None, 64, 64, 3)

			
			return tf.nn.tanh(conv4_batch_norm)

	@property 
	def vars(self):

		return [var for var in tf.global_variables() if self.name in var.name]


# class Discriminator(object):

# 	def __init__(self, h_dim):
# 		self.x_dim = [None, 64, 64, 3]
# 		self.h_dim = h_dim
# 		self.name = 'dcgan/d_net'

# 	def __call__(self, x, h, training, reuse=True):

# 		with tf.variable_scope(self.name) as vs:
# 			if reuse:
# 				vs.reuse_variables()
# 			# x = placeholder

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 64, 64, 1])

# 			x_h = tf.concat([x, r_h], axis=3)

# 			conv1 = layers.conv2d(
# 				x, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)
# 			# batch_norm
# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)
# 			conv1_a = leaky_relu(conv1_batch_norm)

# 			# conv1_a: (None, 32, 32, 64)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

# 			conv1_a_h = tf.concat([conv1_a, r_h], axis=3)


# 			conv2 = layers.conv2d(
# 				conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = leaky_relu(conv2_batch_norm)

# 			# conv2_a: (None, 16, 16, 128)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

# 			conv2_a_h = tf.concat([conv2_a, r_h], axis=3)

# 			conv3 = layers.conv2d(
# 				conv2_a, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)
# 			conv3_a = leaky_relu(conv3_batch_norm)

# 			# conv3_a: (None, 8, 8, 64 * 4)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

# 			conv3_a_h = tf.concat([conv3_a, r_h], axis=3)

# 			conv4 = layers.conv2d(
# 				conv3_a, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv4_batch_norm = layers.batch_normalization(
# 				conv4, training=training
# 			)


# 			conv4_1 = layers.conv2d(
# 				conv4_batch_norm, 64 * 2, kernel_size=[1, 1], strides=[1, 1],
# 				padding='valid',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv4_1_batch_norm = layers.batch_normalization(
# 				conv4_1, training=training
# 			)

# 			conv4_1_a = leaky_relu(conv4_1_batch_norm)

# 			conv4_2 = layers.conv2d(
# 				conv4_1_a, 64 * 2, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv4_2_batch_norm = layers.batch_normalization(
# 				conv4_2, training=training
# 			)

# 			conv4_2_a = leaky_relu(conv4_2_batch_norm)

# 			conv4_3 = layers.conv2d(
# 				conv4_2_a, 64 * 8, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv4_3_batch_norm = layers.batch_normalization(
# 				conv4_3, training=training
# 			)

	


# 			conv4_a = leaky_relu(conv4_batch_norm + conv4_3_batch_norm)

# 			# conv4_a: (None, 4, 4, 64 * 8)






# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

# 			concat_h = tf.concat([conv4_a, r_h], axis=3)

# 			# concat_h: (None, 4, 4, 64 * 8 + h_dim)

# 			conv5 = layers.conv2d(
# 				concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv5_batch_norm = layers.batch_normalization(
# 				conv5, training=training
# 			)
# 			conv5_a = leaky_relu(conv5_batch_norm)

# 			# conv5_a: (None, 4, 4, 64 * 8)

# 			# conv6 = layers.conv2d(
# 			# 	conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
# 			# 	padding='valid',
# 			# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 			# 	activation=None
# 			# )

# 			# return conv6


# 			conv6 = layers.conv2d(
# 				conv5_a, self.h_dim, kernel_size=[4, 4], strides=[1, 1],
# 				padding='valid',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# conv6: (None, 1, 1, 225)


# 			conv6_sq = tf.squeeze(conv6, [1, 2])

# 			# conv6_sq: (None, 225)

# 			# h_h = tf.reshape(tf.matmul(h[:, :15, None], h[:, None, 15:]), [-1, 15 * 15])


# 			fetch = tf.reduce_sum(tf.multiply(conv6_sq, h), axis=1)

# 			# # fetch: (None)

# 			return fetch





# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]


# class Generator(object):

# 	def __init__(self, z_dim, h_dim):
# 		self.z_dim = z_dim
# 		self.h_dim = h_dim
# 		self.x_dim = [None, 64, 64, 3]
# 		self.name = 'dcgan/g_net'

# 	def __call__(self, z, h, training):
		
# 		with tf.variable_scope(self.name) as vs:

# 			# concat z, h
# 			z_h = tf.concat([z, h], axis=1)

# 			# fc
# 			fc1 = layers.dense(
# 				z_h, 64 * 8 * 4 * 4,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_batch_norm = layers.batch_normalization(
# 				fc1, training=training
# 			)

# 			fc1_conv = tf.reshape(fc1_batch_norm, [-1, 4, 4, 64 * 8])

# 			fc1_1conv = layers.conv2d(
# 				fc1_conv, 64 * 2, kernel_size=[1, 1], strides=[1, 1],
# 				padding='valid',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_1conv_batch_norm = layers.batch_normalization(
# 				fc1_1conv, training=training
# 			)

# 			fc1_1conv_a = tf.nn.relu(fc1_1conv_batch_norm)

# 			fc1_2conv = layers.conv2d(
# 				fc1_1conv_a, 64 * 2, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_2conv_batch_norm = layers.batch_normalization(
# 				fc1_2conv, training=training
# 			)

# 			fc1_2conv_a = tf.nn.relu(fc1_2conv_batch_norm)

# 			fc1_3conv = layers.conv2d(
# 				fc1_2conv_a, 64 * 8, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_3conv_batch_norm = layers.batch_normalization(
# 				fc1_3conv, training=training
# 			)


# 			fc1_a = tf.nn.relu(fc1_conv + fc1_3conv_batch_norm)

# 			# fc1_a: (None, 64 * 8 * 4 * 4)

# 			# conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])
# 			conv = fc1_a

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

# 			conv_h = tf.concat([conv, r_h], axis=3)

# 			conv1 = layers.conv2d_transpose(
# 				conv_h, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)



# 			conv1_1conv = layers.conv2d(
# 				conv1_batch_norm, 64, kernel_size=[1, 1], strides=[1, 1],
# 				padding='valid',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_1conv_batch_norm = layers.batch_normalization(
# 				conv1_1conv, training=training
# 			)

# 			conv1_1conv_a = tf.nn.relu(conv1_1conv_batch_norm)

# 			conv1_2conv = layers.conv2d(
# 				conv1_1conv_a, 64, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_2conv_batch_norm = layers.batch_normalization(
# 				conv1_2conv, training=training
# 			)

# 			conv1_2conv_a = tf.nn.relu(conv1_2conv_batch_norm)

# 			conv1_3conv = layers.conv2d(
# 				conv1_2conv_a, 64 * 4, kernel_size=[3, 3], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_3conv_batch_norm = layers.batch_normalization(
# 				conv1_3conv, training=training
# 			)

# 			conv1_a = tf.nn.relu(conv1_batch_norm + conv1_3conv_batch_norm)

# 			# conv1_a: (None, 8, 8, 64 * 4)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

# 			conv1_a_h = tf.concat([conv1_a, r_h], axis=3)

# 			conv2 = layers.conv2d_transpose(
# 				conv1_a_h, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = tf.nn.relu(conv2_batch_norm)

# 			# conv2_a: (None, 16, 16, 64 * 2)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

# 			conv2_a_h = tf.concat([conv2_a, r_h], axis=3)


# 			conv3 = layers.conv2d_transpose(
# 				conv2_a_h, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)
# 			conv3_a = tf.nn.relu(conv3_batch_norm)

# 			# conv3: (None, 32, 32, 64)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

# 			conv3_a_h = tf.concat([conv3_a, r_h], axis=3)			

# 			conv4 = layers.conv2d_transpose(
# 				conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv4_batch_norm = layers.batch_normalization(
# 				conv4, training=training
# 			)

# 			# conv4: (None, 64, 64, 3)

# 			return tf.nn.tanh(conv4_batch_norm)

# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]

# # try dcgan
# class Discriminator(object):

# 	def __init__(self, h_dim):
# 		self.x_dim = [None, 64, 64, 3]
# 		self.h_dim = h_dim
# 		self.name = 'dcgan/d_net'

# 	def __call__(self, x, h, training, reuse=True):

# 		with tf.variable_scope(self.name) as vs:
# 			if reuse:
# 				vs.reuse_variables()
# 			# x = placeholder

# 			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 64, 64, 1])

# 			# x_h = tf.concat([x, r_h], axis=3)

# 			conv1 = layers.conv2d(
# 				x, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)
# 			# batch_norm
# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)
# 			conv1_a = leaky_relu(conv1_batch_norm)

# 			# conv1_a: (None, 32, 32, 64)

# 			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

# 			# conv1_a_h = tf.concat([conv1_a, r_h], axis=3)


# 			conv2 = layers.conv2d(
# 				conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = leaky_relu(conv2_batch_norm)

# 			# conv2_a: (None, 16, 16, 128)

# 			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

# 			# conv2_a_h = tf.concat([conv2_a, r_h], axis=3)

# 			conv3 = layers.conv2d(
# 				conv2_a, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)
# 			conv3_a = leaky_relu(conv3_batch_norm)

# 			# conv3_a: (None, 8, 8, 64 * 4)

# 			# r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

# 			# conv3_a_h = tf.concat([conv3_a, r_h], axis=3)

# 			conv4 = layers.conv2d(
# 				conv3_a, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv4_batch_norm = layers.batch_normalization(
# 				conv4, training=training
# 			)
# 			conv4_a = leaky_relu(conv4_batch_norm)

# 			# conv3_a: (None, 4, 4, 64 * 8)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

# 			concat_h = tf.concat([conv4_a, r_h], axis=3)

# 			# concat_h: (None, 4, 4, 64 * 8 + h_dim)

# 			conv5 = layers.conv2d(
# 				concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
# 				padding='same',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv5_batch_norm = layers.batch_normalization(
# 				conv5, training=training
# 			)
# 			conv5_a = leaky_relu(conv5_batch_norm)

# 			# conv5_a: (None, 4, 4, 64 * 8)

# 			conv6 = layers.conv2d(
# 				conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
# 				padding='valid',
# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			return conv6

# 			# conv6 = layers.conv2d(
# 			# 	conv5_a, self.h_dim, kernel_size=[4, 4], strides=[1, 1],
# 			# 	padding='valid',
# 			# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
# 			# 	activation=None
# 			# )

# 			# # conv6: (None, 1, 1, 225)



# 			# conv6_sq = tf.squeeze(conv6, [1, 2])

# 			# # conv6_sq: (None, 225)

# 			# # h_h = tf.reshape(tf.matmul(h[:, :15, None], h[:, None, 15:]), [-1, 15 * 15])


# 			# fetch = tf.reduce_sum(tf.multiply(conv6_sq, h), axis=1)

# 			# # fetch: (None)

# 			# return fetch





# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]


# class Generator(object):

# 	def __init__(self, z_dim, h_dim):
# 		self.z_dim = z_dim
# 		self.h_dim = h_dim
# 		self.x_dim = [None, 64, 64, 3]
# 		self.name = 'dcgan/g_net'

# 	def __call__(self, z, h, training):
		
# 		with tf.variable_scope(self.name) as vs:

# 			# concat z, h
# 			z_h = tf.concat([z, h], axis=1)

# 			# fc
# 			fc1 = layers.dense(
# 				z_h, 64 * 8 * 4 * 4,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_batch_norm = layers.batch_normalization(
# 				fc1, training=training
# 			)

# 			fc1_a = tf.nn.relu(fc1_batch_norm)

# 			# fc1_a: (None, 64 * 8 * 4 * 4)

# 			conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

# 			conv_h = tf.concat([conv, r_h], axis=3)

# 			conv1 = layers.conv2d_transpose(
# 				conv_h, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)
# 			conv1_a = tf.nn.relu(conv1_batch_norm)

# 			# conv1_a: (None, 8, 8, 64 * 4)
# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 8, 8, 1])

# 			conv1_a_h = tf.concat([conv1_a, r_h], axis=3)

# 			conv2 = layers.conv2d_transpose(
# 				conv1_a_h, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = tf.nn.relu(conv2_batch_norm)

# 			# conv2_a: (None, 16, 16, 64 * 2)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 16, 16, 1])

# 			conv2_a_h = tf.concat([conv2_a, r_h], axis=3)


# 			conv3 = layers.conv2d_transpose(
# 				conv2_a_h, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)
# 			conv3_a = tf.nn.relu(conv3_batch_norm)

# 			# conv3: (None, 32, 32, 64)

# 			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 32, 32, 1])

# 			conv3_a_h = tf.concat([conv3_a, r_h], axis=3)			

# 			conv4 = layers.conv2d_transpose(
# 				conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv4_batch_norm = layers.batch_normalization(
# 				conv4, training=training
# 			)

# 			# conv4: (None, 64, 64, 3)

# 			return tf.nn.tanh(conv4_batch_norm)

# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]


# try dcgan
class Discriminator(object):

	def __init__(self, h_dim):
		self.x_dim = [None, 64, 64, 3]
		self.h_dim = h_dim
		self.name = 'dcgan/d_net'

	def __call__(self, x, h, training, reuse=True):

		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
			# x = placeholder

			conv1 = layers.conv2d(
				x, 64, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)
			# batch_norm
			conv1_batch_norm = layers.batch_normalization(
				conv1, training=training
			)
			conv1_a = leaky_relu(conv1_batch_norm)

			# conv1_a: (None, 32, 32, 64)

			conv2 = layers.conv2d(
				conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv2_batch_norm = layers.batch_normalization(
				conv2, training=training
			)
			conv2_a = leaky_relu(conv2_batch_norm)

			# conv2_a: (None, 16, 16, 128)

			conv3 = layers.conv2d(
				conv2_a, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv3_batch_norm = layers.batch_normalization(
				conv3, training=training
			)
			conv3_a = leaky_relu(conv3_batch_norm)

			# conv3_a: (None, 8, 8, 64 * 4)

			conv4 = layers.conv2d(
				conv3_a, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv4_batch_norm = layers.batch_normalization(
				conv4, training=training
			)
			conv4_a = leaky_relu(conv4_batch_norm)

			# conv3_a: (None, 4, 4, 64 * 8)

			r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

			concat_h = tf.concat([conv4_a, r_h], axis=3)

			# concat_h: (None, 4, 4, 64 * 8 + h_dim)

			conv5 = layers.conv2d(
				concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
				padding='same',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# batch_norm
			conv5_batch_norm = layers.batch_normalization(
				conv5, training=training
			)
			conv5_a = leaky_relu(conv5_batch_norm)

			# conv5_a: (None, 4, 4, 64 * 8)

			conv6 = layers.conv2d(
				conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
				padding='valid',
				kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
				activation=None
			)

			# conv6: (None, 1, 1, 1)

			conv6_sq = tf.squeeze(conv6, [1, 2, 3])

			return conv6_sq





	@property 
	def vars(self):

		return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):

	def __init__(self, z_dim, h_dim, generator_output_layer):
		self.z_dim = z_dim
		self.h_dim = h_dim
		self.x_dim = [None, 64, 64, 3]
		self.name = 'dcgan/g_net'
		self.generator_output_layer = generator_output_layer

	def __call__(self, z, h, training):
		
		with tf.variable_scope(self.name) as vs:

			# concat z, h
			z_h = tf.concat([z, h], axis=1)

			# fc
			fc1 = layers.dense(
				z_h, 64 * 8 * 4 * 4,
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			fc1_batch_norm = layers.batch_normalization(
				fc1, training=training
			)

			fc1_a = tf.nn.relu(fc1_batch_norm)

			# fc1_a: (None, 64 * 8 * 4 * 4)

			conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])

			conv1 = layers.conv2d_transpose(
				conv, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv1_batch_norm = layers.batch_normalization(
				conv1, training=training
			)
			conv1_a = tf.nn.relu(conv1_batch_norm)

			# conv1_a: (None, 8, 8, 64 * 4)

			conv2 = layers.conv2d_transpose(
				conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv2_batch_norm = layers.batch_normalization(
				conv2, training=training
			)
			conv2_a = tf.nn.relu(conv2_batch_norm)

			# conv2_a: (None, 16, 16, 64 * 2)

			conv3 = layers.conv2d_transpose(
				conv2_a, 64, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			conv3_batch_norm = layers.batch_normalization(
				conv3, training=training
			)
			conv3_a = tf.nn.relu(conv3_batch_norm)


			# conv3: (None, 32, 32, 64)

			conv4 = layers.conv2d_transpose(
				conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
			)

			

			# conv4: (None, 64, 64, 3)
			if self.generator_output_layer == 'tanh':
				return tf.nn.tanh(conv4)
			elif self.generator_output_layer == 'sigmoid':
				return tf.nn.sigmoid(conv4)

	@property 
	def vars(self):

		return [var for var in tf.global_variables() if self.name in var.name]






# class Discriminator(object):

# 	def __init__(self, h_dim):
# 		self.x_dim = [None, 96, 96, 3]
# 		self.h_dim = h_dim
# 		self.name = 'dcgan/d_net'

# 	def __call__(self, x, h, training, reuse=True):

# 		with tf.variable_scope(self.name) as vs:
# 			if reuse:
# 				vs.reuse_variables()
# 			# x = placeholder

# 			conv1 = layers.conv2d(
# 				x, 32, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)
# 			# batch_norm
# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)
# 			conv1_a = leaky_relu(conv1_batch_norm)

# 			# conv1_a: (None, 48, 48, 32)

# 			conv2 = layers.conv2d(
# 				conv1_a, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = leaky_relu(conv2_batch_norm)

# 			# conv2_a: (None, 24, 24, 64)

# 			conv3 = layers.conv2d(
# 				conv2_a, 128, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)
# 			conv3_a = leaky_relu(conv3_batch_norm)

# 			# conv3_a: (None, 12, 12, 128)

# 			fc_h = layers.dense(
# 				h, 300,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)


# 			# flatten
# 			conv_flat = tf.contrib.layers.flatten(conv3_a)

# 			# concat x(from conv) and tag feature h
# 			z_h = tf.concat([conv_flat, fc_h], axis=1)


# 			fc1 = layers.dense(
# 				z_h, 4096,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			fc1_batch_norm = layers.batch_normalization(
# 				fc1, training=training
# 			)
# 			fc1_a = leaky_relu(fc1_batch_norm)


# 			# fully-connected layers
# 			fc2 = layers.dense(
# 				fc1_a, 1024,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# batch_norm
# 			fc2_batch_norm = layers.batch_normalization(
# 				fc2, training=training
# 			)
# 			fc2_a = leaky_relu(fc2_batch_norm)

# 			fc3 = layers.dense(
# 				fc2_a, 1,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			return fc3

# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]


# class Generator(object):

# 	def __init__(self, z_dim, h_dim):
# 		self.z_dim = z_dim
# 		self.h_dim = h_dim
# 		self.x_dim = [None, 96, 96, 3]
# 		self.name = 'dcgan/g_net'

# 	def __call__(self, z, h, training):
		
# 		with tf.variable_scope(self.name) as vs:

# 			fc_h = layers.dense(
# 				h, 300,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			# concat z, h
# 			z_h = tf.concat([z, fc_h], axis=1)

# 			# fc
# 			fc1 = layers.dense(
# 				z_h, 4096,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc1_batch_norm = layers.batch_normalization(
# 				fc1, training=training
# 			)

# 			fc1_a = leaky_relu(fc1_batch_norm)

# 			# fc1_a: (None, 1024)

# 			fc2 = layers.dense(
# 				fc1_a, 12 * 12 * 128,
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			fc2_batch_norm = layers.batch_normalization(
# 				fc2, training=training
# 			)

# 			fc2_a = leaky_relu(fc2_batch_norm)

# 			# fc2_a: (None, 12 * 12 * 128)

# 			conv = tf.reshape(fc2_a, [-1, 12, 12, 128])

# 			conv1 = layers.conv2d_transpose(
# 				conv, 64, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv1_batch_norm = layers.batch_normalization(
# 				conv1, training=training
# 			)
# 			conv1_a = leaky_relu(conv1_batch_norm)

# 			# conv1_a: (None, 24, 24, 64)

# 			conv2 = layers.conv2d_transpose(
# 				conv1_a, 32, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv2_batch_norm = layers.batch_normalization(
# 				conv2, training=training
# 			)
# 			conv2_a = leaky_relu(conv2_batch_norm)

# 			# conv2_a: (None, 48, 48, 32)

# 			conv3 = layers.conv2d_transpose(
# 				conv2_a, 3, kernel_size=[4, 4], strides=[2, 2],
# 				padding='same',
# 				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
# 				activation=None
# 			)

# 			conv3_batch_norm = layers.batch_normalization(
# 				conv3, training=training
# 			)

# 			# conv3: (None, 96, 96, 3)


# 			return tf.nn.sigmoid(conv3_batch_norm)

# 	@property 
# 	def vars(self):

# 		return [var for var in tf.global_variables() if self.name in var.name]















