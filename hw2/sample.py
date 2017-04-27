



class VideoCaptionSample(object):

	def __init__(self, video_id, image_frames, labels):
		self.video_id = video_id
		self.frames = image_frames
		self.labels = labels
		
	# 	self.rewards = self.compute_sequence_bleu(labels, max_doc_len, end_id)


	# def compute_sequence_bleu(self, labels, max_doc_len, vocab_size, end_id):
	# 	pass







		