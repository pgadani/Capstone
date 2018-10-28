import glob
import json
import librosa
import numpy as np
import os
import pickle
import sklearn
import time

from seqlearn.hmm import MultinomialHMM
from sklearn.cluster import KMeans

from viterbi import viterbi


SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second

TOKEN_SIZE = 10 # frames per token

NUM_SKELETON_POS = SAMPLE_LEN * FRAME_RATE
JOINTS = ["head", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank"]


class AudioSample:
	def __init__(self, filename, index, raw, features, cluster=None):
		self.filename = filename
		self.index = index
		self.raw = raw
		self.features = features
		self.cluster = cluster

	def __str__(self):
		return 'Filename: {} \t Index: {} \t Cluster: {}'.format(
				self.filename, self.index, self.cluster)


class MotionToken:
	def __init__(self, filename, person, index, motion_diff, cluster=None):
		self.filename = filename
		self.person = person
		self.index = index
		self.motion_diff = motion_diff
		self.cluster = cluster

	def __str__(self):
		return 'Filename: {} \t Person: {} \t Index: {} \t Cluster: {}'.format(
				self.filename, self.person, self.index, self.cluster)


def load_skeletons(data_path, label='*'):
	skeleton_dir = '{}/{}/{}/*'.format(data_path, 'cplskeleton_final', label)
	skeletons = {}
	for file in glob.glob(skeleton_dir):
		filename = os.path.splitext(os.path.basename(file))[0]
		split_frame_num = filename.rindex('_')
		sample_name = filename[:split_frame_num]
		frame = int(filename[split_frame_num + 1:]) - 1
		if sample_name not in skeletons:
			skeletons[sample_name] = {}
		skeleton = skeletons[sample_name]
		with open(file) as f:
			for line in f:
				data = json.loads(line)
				person = data[0]
				if person not in skeletons[sample_name]:
					skeleton[person] = [{} for i in range(NUM_SKELETON_POS)]
				for position in data[1:]:
					joint, x, y = position
					skeleton[person][frame][joint] = (x, y)
	return skeletons				


def generate_motion_tokens(skeletons):
	tokens = []
	for filename, skel in skeletons.items():
		for person, positions in skel.items():
			motion_diff = []
			for index in range(0, NUM_SKELETON_POS // TOKEN_SIZE - TOKEN_SIZE):
				i = index * TOKEN_SIZE
				has_all_positions = True
				for k in range(TOKEN_SIZE + 1):
					if 'head' not in positions[i + k]:
						has_all_positions = False
				if has_all_positions:
					motion_diff = [positions[i + k + 1][joint][j] - positions[i + k][joint][j] for joint in JOINTS for j in (0, 1) for k in range(TOKEN_SIZE)]
					tokens.append(MotionToken(filename, person, index, motion_diff))
	return tokens


def cluster_motion(tokens, n_clusters):
	motion_diffs = np.array([token.motion_diff for token in tokens])
	kmeans = KMeans(n_clusters=n_clusters).fit(motion_diffs)
	clusters = kmeans.labels_
	for token, label in zip(tokens, clusters):
		token.cluster = label
	return kmeans, clusters


# returns a list of (source file, index, sample, features) tuples
def load_audio(data_path, label='*', sample_time=None):
	if not sample_time:
		sample_time = TOKEN_SIZE / FRAME_RATE
	audio_dir = '{}/{}/{}/*'.format(data_path, 'audio', label)
	audio_samples = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		frame_length = int(sample_time * SAMPLE_RATE)
		audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
		features = []

		audio_chroma = librosa.feature.chroma_cqt(audio_raw, sr=SAMPLE_RATE, hop_length=frame_length)

		mfcc = librosa.feature.mfcc(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length)
		features = np.vstack((audio_chroma, mfcc)).T

		print(audio_file, features.shape, mfcc.shape, audio_chroma.shape)
		for index, feature in enumerate(features):
			audio_samples.append(AudioSample(filename, index, None, feature))
	return audio_samples


def cluster_audio(audio_samples, n_clusters):
	audio_features = np.array([sample.features for sample in audio_samples])
	kmeans = KMeans(n_clusters=n_clusters).fit(audio_features)
	clusters = kmeans.labels_
	sequences = []
	for sample, cluster in zip(audio_samples, clusters):
		sample.cluster = cluster
	return kmeans, clusters


# Generates emission probability matrix
# Probability of observing audio from state of motion
def emission_probability(audio_samples_map, audio_clusters, motion_tokens, motion_clusters):
	transitions = np.zeros((audio_clusters, motion_clusters))
	for token in motion_tokens:
		if (token.filename, token.index) in audio_samples_map:
			sample = audio_samples_map[(token.filename, token.index)]
			transitions[sample.cluster, token.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


def transition_probability(audio_samples, audio_clusters):
	transitions = np.zeros((audio_clusters, audio_clusters))
	for prev, curr in zip(audio_samples[:-1], audio_samples[1:]):
		if prev.filename != curr.filename:
			continue
		transitions[prev.cluster][curr.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


def main(pickle_data=True, label='*', audio_clusters=25, motion_clusters=25):
	rewrite = True
	pickle_samples = 'pickles/audio_{}.pkl'.format(label)
	audio_samples = means = None
	if pickle_data:
		if os.path.exists(pickle_samples):
			with open(pickle_samples, 'rb') as f:
				audio_samples = pickle.load(f)
			rewrite = False
	if audio_samples is None:
		audio_samples = load_audio('..', label=label)
	if pickle_data and rewrite:
		with open(pickle_samples, 'wb+') as f:
			pickle.dump(audio_samples, f)


	pickle_skeletons = 'pickles/skeleton_{}.pkl'.format(label)
	pickle_motion_tok = 'pickles/motion_tokens_{}.pkl'.format(label)
	skeletons = None
	motion_tokens = None
	if pickle_data:
		if os.path.exists(pickle_skeletons):
			with open(pickle_skeletons, 'rb') as f:
				skeletons = pickle.load(f)
		if os.path.exists(pickle_motion_tok):
			with open(pickle_motion_tok, 'rb') as f:
				motion_tokens = pickle.load(f)

	if not skeletons:
		skeletons = load_skeletons('..', label=label)
		if pickle_data:
			with open(pickle_skeletons, 'wb+') as f:
				pickle.dump(skeletons, f)

	if not motion_tokens:
		motion_tokens = generate_motion_tokens(skeletons)
		if pickle_data:
			with open(pickle_motion_tok, 'wb+') as f:
				pickle.dump(motion_tokens, f)

	audio_classifier, audio_labels = cluster_audio(audio_samples, audio_clusters)
	motion_classifier, motion_labels = cluster_motion(motion_tokens, motion_clusters)

	# print('AUDIO SAMPLES')
	# for sample in audio_samples:
	# 	print(sample)

	audio_map = {(sample.filename, sample.index):sample for sample in audio_samples}

	motion_tokens.sort(key=lambda tok: (tok.filename, tok.person, tok.index))
	audio_labels_seq = []
	motion_labels_seq = []
	motion_tokens = list(filter(lambda tok: (tok.filename, tok.index) in audio_map, motion_tokens))


	# for tok in motion_tokens:
	# 	print(tok)
	motion_labels_seq = [tok.cluster for tok in motion_tokens]
	audio_labels_seq = np.array([audio_map[(tok.filename, tok.index)].cluster for tok in motion_tokens])

	sequence_lens = []
	curr_len = 1
	for prev, curr in zip(motion_tokens[:-1], motion_tokens[1:]):
		if curr.filename != prev.filename or curr.person != prev.person or curr.index - prev.index != 1:
			sequence_lens.append(curr_len)
			curr_len = 0
	sequence_lens.append(curr_len)

	# Make an actual label binarizer object
	motion_encoded = sklearn.preprocessing.label_binarize(motion_labels_seq, classes=list(range(motion_clusters)))

	# transition_prob = transition_probability(audio_samples, audio_clusters)
	# emission_prob = emission_probability(audio_map, audio_clusters, motion_tokens, motion_clusters)

	print('MOTION ONE-HOT SHAPE', motion_encoded.shape)
	print('AUDIO LABELS SHAPE', audio_labels_seq.shape)
	print('SEQUENCE LENS SUM', sum(sequence_lens))

	hmm = MultinomialHMM()
	hmm.fit(motion_encoded, audio_labels_seq, sequence_lens)


	all_samples = {(tok.filename, tok.index, tok.person) for tok in motion_tokens}
	recon_samples = []

	for samp in all_samples:
		audio_samp = (samp[0], samp[1])
		if audio_samp in audio_map:
			recon_samples.append(samp)
			if len(recon_samples) >= 10:
				break


	for samp in recon_samples:
		print(samp[0], samp[1])
		recon_motion = list(filter(lambda tok: tok.filename == samp[0] and tok.person == samp[2], motion_tokens))
		recon_motion.sort(key=lambda tok: tok.index)

		expected_audio = [audio_map[(tok.filename, tok.index)] for tok in recon_motion]

		# result = viterbi([tok.cluster for tok in recon_motion], transition_prob, emission_prob)
		
		recon_motion_encoded = sklearn.preprocessing.label_binarize([tok.cluster for tok in recon_motion], classes=list(range(motion_clusters)))

		result = hmm.predict(recon_motion_encoded)

		for tok, audio in zip(recon_motion, expected_audio):
			if tok.filename != audio.filename or tok.index != audio.index:
				print("MISMATCHED", tok.filename, tok.index, audio.filename, audio.index)

		errors = 0
		for i, cluster in enumerate(result):
			print(cluster, expected_audio[i].cluster)
			if cluster != expected_audio[i].cluster:
				errors += 1

		print("ERROR: ", errors / len(result))




if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label='swing')
	print("This took {:4.2f}s".format(time.time()-start_time))