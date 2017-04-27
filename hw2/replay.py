import math
import random
import bisect

class Sample(object):
	def __init__(self, 
		frames, 
		input_c_state, 
		input_h_state, 
		output_c_state, 
		output_h_state, 
		action, 
		reward, 
		terminal):
		self.frames = frames
		self.input_c_state = input_c_state
		self.input_h_state = input_h_state
		self.output_c_state = output_c_state
		self.output_h_state = output_h_state
		self.action = action
		self.reward = reward
		self.terminal = terminal

		self.weight = 1
		self.cumulative_weight = 1
	def is_interesting(self):
		return self.terminal or self.reward != 0

	def __lt__(self, obj):
		return self.cumulative_weight - obj.cumulative_weight

class ReplayMemory(object):
	def __init__(self, replay_capacity, prioritized_replay):
		self.samples = []
		self.max_samples = replay_capacity
		self.prioritized_replay = prioritized_replay
		self.num_interesting_samples = 0
		self.batches_drawn = 0

		self.index = 0

	def num_samples(self):
		return len(self.samples)

	def add_sample(self, sample):
		self.samples.append(sample)
		self._update_weights_for_newly_added_sample()
		# self._truncate_list_if_necessary()


	def _update_weights_for_newly_added_sample(self):

		if len(self.samples) > 1:
			self.samples[-1].cumulative_weight = self.samples[-1].weight + self.samples[-2].cumulative_weight

		if self.samples[-1].is_interesting():
			self.num_interesting_samples += 1

			# Boost the neighboring samples.  How many samples?  Roughly the number of samples
			# that are "uninteresting".  Meaning if interesting samples occur 3% of the time, then boost 33

			uninteresting_smaple_range = int(max(1, len(self.samples) / max(1, self.num_interesting_samples)))
			for i in range(uninteresting_smaple_range, 0, -1):
				index = len(self.samples) - i
				if index < 1:
					break
				# This is an exponential that ranges from 3.0 to 1.01 over the domain of [0, uninteresting_sample_range ]
				# So the interesting sample gets a 3x boost, and the one furthest away gets a 1% boost
				boost = 1.0 + 3.0/(math.exp(i/(uninteresting_smaple_range/6.0)))
				self.samples[index].weight *= boost
				self.samples[index].cumulative_weight = self.samples[index].weight + self.samples[index - 1].cumulative_weight

	def _truncate_list_if_necessary(self):
		# premature optimizastion alert :-), don't truncate on each
		# added sample since (I assume) it requires a memcopy of the list (probably 8mb)
		if len(self.samples) > self.max_samples * 1.20:
			truncated_weight = 0
			# Before truncating the list, correct self.numInterestingSamples, and prepare
			# for correcting the cumulativeWeights of the remaining samples
			for i in range(self.max_samples, len(self.samples)):
				truncated_weight += self.samples[i].weight
				if self.samples[i].is_interesting():
					self.num_interesting_samples -= 1

			# Truncate the list
			self.samples = self.samples[(len(self.samples) - self.max_samples):]
			
			# Correct cumulativeWeights
			for sample in self.samples:
				sample.cumulative_weight -= truncated_weight

	def truncate_list_if_necessary(self):
		self._truncate_list_if_necessary()

	def draw_batch(self, batch_size):
		if batch_size > len(self.samples):
			raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batch_size))

		self.batches_drawn += 1

		if self.prioritized_replay:
			return self._draw_prioritized_batch(batch_size)
		else:
			samples = self.samples[self.index : self.index + batch_size]
			if self.index + batch_size >= self.num_samples():
				self.index = 0
			else:
				self.index += batch_size
			return samples
			# return random.sample(self.samples, batch_size)

	# The nature paper doesn't do this but they mention the idea.
	# This particular approach and the weighting I am using is a total
	# uninformed fabrication on my part.  There is probably a more
	# principled way to do this

	def _draw_prioritized_batch(self, batch_size):
		batch = []
		probe = Sample(None, None, None, None, None, 0, 0, False)
		while len(batch) < batch_size:
			probe.cumulative_weight = random.uniform(0, self.samples[-1].cumulative_weight)
			index = bisect.bisect_right(self.samples, probe, 0, len(self.samples) - 1)
			sample = self.samples[index]
			sample.weight = max(1, .8 * sample.weight)
			if sample not in batch:
				batch.append(sample)

		if self.batches_drawn % 100 == 0:
			cumulative = 0
			for sample in self.samples:
				cumulative += sample.weight
				sample.cumulative_weight = cumulative
		return batch
	
	def _print_batch_weight(self, batch):
		batch_weight = 0
		for i in range(0, len(batch)):
			batch_weight += batch[i].weight
		print('batch weight: %f' % batch_weight)















