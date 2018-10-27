import glob
import json
import librosa
import numpy as np
import os
import pickle
import time
import sklearn
from sklearn.cluster import KMeans

from viterbi import viterbi

SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second
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
		return 'Filename: {} \t Person: {} \t Index: {} \t Motion Diff: {} \t Cluster: {}'.format(
				self.filename, self.person, self.index, self.motion_diff, self.cluster)


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
			for i in range(1, NUM_SKELETON_POS):
				new_pos = positions[i]
				old_pos = positions[i - 1]
				if new_pos and old_pos:
					motion_diff = [new_pos[joint][i] - old_pos[joint][i] for joint in JOINTS for i in (0, 1)]
					tokens.append(MotionToken(filename, person, i, motion_diff))
	return tokens


def cluster_motion(tokens, n_clusters):
	motion_diffs = np.array([token.motion_diff for token in tokens])
	kmeans = KMeans(n_clusters=n_clusters).fit(motion_diffs)
	clusters = kmeans.labels_
	print(clusters)
	for token, label in zip(tokens, clusters):
		token.cluster = label
	means = {i: a for i, a in enumerate(kmeans.cluster_centers_)}
	return clusters, means


# returns a list of (source file, index, sample, features) tuples
def load_audio(data_path, label='*', sample_time=None):
	if not sample_time:
		sample_time = 1 / FRAME_RATE
	audio_dir = '{}/{}/{}/*'.format(data_path, 'audio', label)
	audio_samples = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		print(audio_file)
		for index, offset in enumerate(np.arange(0, file_duration, sample_time)):
			audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE, offset=offset, duration=sample_time)
			audio_chroma = librosa.feature.chroma_stft(audio_raw, sr=SAMPLE_RATE)
			features = np.ndarray.flatten(audio_chroma)
			print(index, len(features), filename)
			audio_samples.append(AudioSample(filename, index, audio_raw, features))
	return audio_samples


def cluster_audio(audio_samples, n_clusters):
	audio_features = np.array([sample.features for sample in audio_samples])
	print(audio_features.shape)
	kmeans = KMeans(n_clusters=n_clusters).fit(audio_features)
	clusters = kmeans.labels_
	print(clusters)
	means = {i: a for i, a in enumerate(kmeans.cluster_centers_)}
	return clusters, means


# Generates emission probability matrix
# Probability of observing audio from state of motion
def probability_motion_from_audio(audio_samples_map, audio_clusters, motion_tokens, motion_clusters):
	transitions = np.zeros((audio_clusters, motion_clusters))
	for token in motion_tokens:
		if (token.filename, token.index) in audio_samples_map:
			sample = audio_samples_map[(token.filename, token.index)]
			transitions[sample.cluster, token.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	for row in probabilities:
		print(row)
	return probabilities



def main(pickle_data=True, label='*', n_clusters=32):
	rewrite = True
	pickle_samples = 'pickles/audio_{}_{}.pkl'.format(label, n_clusters)
	pickle_means = 'pickles/means_{}_{}.pkl'.format(label, n_clusters)
	audio_samples = means = None
	if pickle_data:
		if os.path.exists(pickle_samples) and os.path.exists(pickle_means):
			with open(pickle_samples, 'rb') as f:
				audio_samples = pickle.load(f)
			with open(pickle_means, 'rb') as f:
				means = pickle.load(f)
			rewrite = False
	if audio_samples is None or means is None:
		audio_samples = load_audio('..', label=label)
		clusters, means = cluster_audio(audio_samples, n_clusters)
		for sample, cluster in zip(audio_samples, clusters):
			sample.cluster = cluster
	if pickle_data and rewrite:
		with open(pickle_samples, 'wb+') as f:
			pickle.dump(audio_samples, f)
		with open(pickle_means, 'wb+') as f:
			pickle.dump(means, f)
	transition_prob = np.zeros((n_clusters, n_clusters))
	for prev, curr in zip(audio_samples[:-1], audio_samples[1:]):
		if prev.filename != curr.filename:
			continue
		transition_prob[prev.cluster][curr.cluster] += 1
	print(transition_prob)
	row_sums = transition_prob.sum(axis=1, keepdims=True)
	transition_prob = transition_prob / row_sums


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
		skeletons = load_skeletons('..', label='ballet')
		if pickle_data:
			with open(pickle_skeletons, 'wb+') as f:
				pickle.dump(skeletons, f)

	if not motion_tokens:
		motion_tokens = generate_motion_tokens(skeletons)
		cluster_motion(motion_tokens, n_clusters)
		if pickle_data:
			with open(pickle_motion_tok, 'wb+') as f:
				pickle.dump(motion_tokens, f)


	audio_map = {(sample.filename, sample.index):sample for sample in audio_samples}
	emission_prob = probability_motion_from_audio(audio_map, n_clusters, motion_tokens, n_clusters)

	recon_tok = None
	recon_audio = None
	for token in motion_tokens:
		if (token.filename, token.index) in audio_map:
			recon_tok = token
			recon_audio = audio_map[(token.filename, token.index)]

	recon_motion = list(filter(lambda tok: tok.filename == recon_tok.filename and tok.person == recon_tok.person, motion_tokens))
	recon_motion.sort(key=lambda tok: tok.index)

	expected_audio = [audio_map[(tok.filename, tok.index)] for tok in recon_motion]

	result = viterbi([tok.cluster for tok in recon_motion], transition_prob, emission_prob)

	for tok, samp in zip(recon_motion, expected_audio):
		if tok.filename != samp.filename or tok.index != samp.index:
			print("MISMATCHED", tok.filename, tok.index, samp.filename, samp.index)

	errors = 0
	for i, cluster in enumerate(result):
		print(cluster, expected_audio[i].cluster)
		if cluster != expected_audio[i].cluster:
			errors += 1

	print("ERROR: ", errors / len(result))



if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label='ballet', n_clusters=48)
	print("This took {:4.2f}s".format(time.time()-start_time))