import argparse
import glob
import json
import librosa
import math
import numpy as np
import os
import pickle
import random
import sklearn
import time

from matplotlib import pyplot as plt
from seqlearn.hmm import MultinomialHMM
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from viterbi import viterbi


LABEL = 'swing'
CLUSTERING = 'k' # k for kmeans, m for meanshift

SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second

TOKEN_SIZE = 10 # frames per token

MOTION_STRIDE = 1 # default is 1
MOTION_LENGTH = 10 # default is TOKEN_SIZE // MOTION_STRIDE

N_CHROMA = 4 # default 12
N_MFCC = 4 # default 20
N_TEMPO = 3 # don't know default

N_MUSIC_CLUSTERS = 12
N_MOTION_CLUSTERS = 15 # 12

NUM_SKELETON_POS = SAMPLE_LEN * FRAME_RATE
JOINTS = ["head", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank"]
JOINTS_DISPLAY = ["head", "neck", \
					"Rsho", "Relb", "Rwri", "Relb", "Rsho", "neck", \
					"Lsho", "Lelb", "Lwri", "Lelb", "Lsho", "neck", \
					"Rhip", "Rkne", "Rank", "Rkne", "Rhip", \
					"Lhip", "Lkne", "Lank", "Lkne", "Lhip", "neck"]

class AudioToken:
	def __init__(self, filename, index, cluster=None):
		self.filename = filename
		self.index = index
		self.cluster = cluster

	def __str__(self):
		return 'Filename: {} \t Index: {} \t Cluster: {}'.format(
				self.filename, self.index, self.cluster)


class MotionToken:
	def __init__(self, filename, person, index, skeletons, cluster=None):
		self.filename = filename
		self.person = person
		self.index = index
		self.skeletons = skeletons
		self.cluster = cluster

	def __str__(self):
		return 'Filename: {} \t Person: {} \t Index: {} \t Cluster: {}'.format(
				self.filename, self.person, self.index, self.cluster)


def print_float_2d(arr, dec=2):
	format_str = '{{:0.{}f}}'.format(dec)
	for row in arr:
		for val in row:
			print(format_str.format(val), end=' ')
		print()


def load_skeletons(data_path, label='*'):
	skeleton_dir = '{}/{}/{}/*'.format(data_path, 'skeletons_cleaned', label)
	skeletons = {}
	for file in glob.glob(skeleton_dir):
		filename = os.path.splitext(os.path.basename(file))[0]
		if filename not in skeletons:
			skeletons[filename] = {}
		skeleton = skeletons[filename]
		with open(file) as f:
			data = json.load(f)
			for frame, skels in enumerate(data):
				for person, joints in skels.items():
					if person not in skeleton:
						skeleton[person] = [{} for i in range(NUM_SKELETON_POS)]
					for joint, pos in joints.items():
						skeleton[person][frame][joint] = pos
			for person in list(skeleton.keys()):
				if len([frame for frame in skeleton[person] if frame]) < 0.1 * NUM_SKELETON_POS:
					del skeleton[person]
	return skeletons


def transform(point, offset, angle, scale):
	new_point = [(p - o) * scale for p, o in zip(point, offset)]
	if angle:
		new_point = [new_point[0] * math.cos(angle) - new_point[1] * math.sin(angle), new_point[1] * math.cos(angle) + new_point[0] * math.sin(angle)]
	return new_point


def generate_motion_tokens(skeletons):
	tokens = []
	n_frames = min(TOKEN_SIZE, MOTION_STRIDE * MOTION_LENGTH)
	all_features = []
	for filename, skel in skeletons.items():
		for person, positions in skel.items():
			for index in range(0, NUM_SKELETON_POS // TOKEN_SIZE - TOKEN_SIZE):
				i = index * TOKEN_SIZE
				has_all_positions = True
				for k in range(0, n_frames + 1, MOTION_STRIDE):
					if 'head' not in positions[i + k]:
						has_all_positions = False
						break
				if not has_all_positions:
					break
				# use center of hips as origin
				offset = [(positions[i]['Lhip'][j] + positions[i]['Rhip'][j])/2 for j in range(2)]
				# scale based on neck being one unit away
				diff = [positions[i]['neck'][j] - offset[j] for j in range(2)]
				# not messing with angle for now
				# angle = math.atan2(*diff)
				angle = 0
				scale = 1/math.sqrt(diff[0]**2 + diff[1]**2)
				motion_features = []
				token_skels = []
				for k in range(i, i + n_frames + 1 - MOTION_STRIDE, MOTION_STRIDE):
					curr_skel = {}
					for joint in JOINTS:
						curr_skel[joint] = transform(positions[k][joint], offset, angle, scale)
					token_skels.append(curr_skel)
				for s in token_skels:
					for j1 in JOINTS:
						for j2 in JOINTS:
							if j1 == j2:
								continue
							motion_features += [s[j1][i] - s[j2][i] for i in range(2)]
				for s, n in zip(token_skels[:-1], token_skels[1:]):
					for j1 in JOINTS:
						for j2 in JOINTS:
								motion_features += [s[j1][i] - n[j2][i] for i in range(2)]
				# for s in token_skels[1:]:
				# 	s0 = token_skels[0]
				# 	for j1 in JOINTS:
				# 		for j2 in JOINTS:
				# 				motion_features += [s[j1][i] - s0[j2][i] for i in range(2)]
				all_features.append(motion_features)
				tokens.append(MotionToken(filename, person, index, token_skels))
	features = np.array(all_features)
	features = normalize(features, norm='max')
	pca = PCA(n_components=.95)
	features = pca.fit_transform(features)
	return tokens, features


def cluster_motion(tokens, features, n_clusters):
	if CLUSTERING == 'm':
		# default bandwidth around 32
		classifier = MeanShift(cluster_all=False).fit(features)
		print(classifier.get_params())
		print(estimate_bandwidth(features))
		n_clusters = len(classifier.cluster_centers_)
	else:
		classifier = KMeans(n_clusters=n_clusters).fit(features)

	clusters = classifier.labels_
	for token, label in zip(tokens, clusters):
		token.cluster = label
	return classifier, clusters, n_clusters


# returns a list of (source file, index, sample, features) tuples
def load_audio(data_path, label='*', sample_time=None):
	if not sample_time:
		sample_time = TOKEN_SIZE / FRAME_RATE
	audio_dir = '{}/{}/{}/*'.format(data_path, 'audio', label)
	audio_tokens = []
	all_features = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		frame_length = int(sample_time * SAMPLE_RATE)
		audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
		features = []

		audio_chroma = librosa.feature.chroma_cqt(audio_raw, sr=SAMPLE_RATE, hop_length=frame_length, n_chroma=N_CHROMA)
		mfcc = librosa.feature.mfcc(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length, n_mfcc=N_MFCC)
		mfcc_diff = librosa.feature.delta(mfcc)
		tempo = librosa.feature.tempogram(audio_raw, sr=SAMPLE_RATE, hop_length=frame_length, win_length=N_TEMPO)
		onset = librosa.onset.onset_strength(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length)
		onset = np.reshape(onset, (1, len(onset)))

		features = np.vstack((audio_chroma, mfcc, mfcc_diff, tempo, onset)).T

		print(audio_file, features.shape, mfcc.shape, audio_chroma.shape, tempo.shape, mfcc_diff.shape)
		for index, feature in enumerate(features):
			audio_tokens.append(AudioToken(filename, index))
			all_features.append(feature)
	return audio_tokens, all_features


def cluster_audio(audio_tokens, audio_features, n_clusters):
	# print('audio', audio_features.shape)
	# pca = PCA(n_components=.9)
	# audio_features = pca.fit_transform(audio_features)
	# print(audio_features.shape)

	# classifier = MeanShift().fit(audio_features)
	# n_clusters = len(classifier.cluster_centers_)
	classifier = KMeans(n_clusters=n_clusters).fit(audio_features)
	clusters = classifier.labels_
	sequences = []
	for tok, cluster in zip(audio_tokens, clusters):
		tok.cluster = cluster
	return classifier, clusters, n_clusters


# Generates emission probability matrix
# Probability of observing audio from state of motion
def emission_probability(audio_tok_map, audio_clusters, motion_tokens, motion_clusters):
	transitions = np.zeros((audio_clusters, motion_clusters))
	for motion_token in motion_tokens:
		if (motion_token.filename, motion_token.index) in audio_tok_map:
			audio_token = audio_tok_map[(motion_token.filename, motion_token.index)]
			transitions[audio_token.cluster, motion_token.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


def transition_probability(audio_tokens, audio_clusters):
	transitions = np.zeros((audio_clusters, audio_clusters))
	for prev, curr in zip(audio_tokens[:-1], audio_tokens[1:]):
		if prev.filename != curr.filename:
			continue
		transitions[prev.cluster][curr.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


def token_to_skel(token):
	skeleton = {}
	for i, j in enumerate(JOINTS):
		skeleton[j] = token[2*i:2*i+2]
	return skeleton


def dist(p1, p2):
	return ((p1[0] - p2[0])**2 + (p1[1] + p2[1])**2)**.5


def draw_pose(skel, color='k', subplt=None):
	if subplt:
		subplt.plot(skel['head'][0], -skel['head'][1], 'o')
		subplt.axis('equal')
	else:
		plt.plot(skel['head'][0], -skel['head'][1], 'o')
	# head = plt.Circle((skel['head'][0], -skel['head'][1]), (skel['head'][0]**2 + skel['head'][1]**2)**.5, color=color)
	# plt.gcf().gca().add_artist(head)
	joints_show = JOINTS_DISPLAY
	for j1, j2 in zip(joints_show[:-1], joints_show[1:]):
		p1 = skel[j1]
		p2 = skel[j2]
		if subplt:
			subplt.plot([p1[0], p2[0]], [-p1[1], -p2[1]], color)
		else:
			plt.plot([p1[0], p2[0]], [-p1[1], -p2[1]], color)


def main(pickle_data=True, label='*', audio_clusters=25, motion_clusters=25):
	rewrite = True
	pickle_suffix = '{}'.format(label)

	pickle_motion_tok = 'pickles/motion_tokens_{}.pkl'.format(pickle_suffix)
	pickle_motion_feat = 'pickles/motion_features_{}.pkl'.format(pickle_suffix)
	pickle_audio_tok = 'pickles/audio_tokens_{}.pkl'.format(pickle_suffix)
	pickle_audio_feat = 'pickles/audio_features_{}.pkl'.format(pickle_suffix)
	motion_tokens = None
	motion_features = None
	audio_tokens = None
	audio_features = None

	if pickle_data:
		if os.path.exists(pickle_motion_tok) and os.path.exists(pickle_motion_feat):
			with open(pickle_motion_tok, 'rb') as f:
				motion_tokens = pickle.load(f)
			with open(pickle_motion_feat, 'rb') as f:
				motion_features = pickle.load(f)

	if motion_tokens is None or motion_features is None:
		skeletons = load_skeletons('..', label=label)
		motion_tokens, motion_features = generate_motion_tokens(skeletons)
		if pickle_data:
			with open(pickle_motion_tok, 'wb+') as f:
				pickle.dump(motion_tokens, f)
			with open(pickle_motion_feat, 'wb+') as f:
				pickle.dump(motion_features, f)

	if pickle_data:
		if os.path.exists(pickle_audio_tok) and os.path.exists(pickle_audio_feat):
			with open(pickle_audio_tok, 'rb') as f:
				audio_tokens = pickle.load(f)
			with open(pickle_audio_feat, 'rb') as f:
				audio_features = pickle.load(f)

	if audio_tokens is None or audio_features is None:
		audio_tokens, audio_features = load_audio('..', label=label)
	if pickle_data:
		with open(pickle_audio_tok, 'wb+') as f:
			pickle.dump(audio_tokens, f)
		with open(pickle_audio_feat, 'wb+') as f:
			pickle.dump(audio_features, f)

	audio_classifier, audio_labels, audio_clusters = cluster_audio(audio_tokens, audio_features, audio_clusters)
	motion_classifier, motion_labels, motion_clusters = cluster_motion(motion_tokens, motion_features, motion_clusters)

	audio_cluster_counts = np.zeros(audio_clusters)
	for token in audio_tokens:
		audio_cluster_counts[token.cluster] += 1

	motion_cluster_counts = np.zeros(motion_clusters)
	for token in motion_tokens:
		motion_cluster_counts[token.cluster] += 1

	motion_tokens.sort(key=lambda tok: (tok.filename, tok.person, tok.index))

	print("AUDIO CLUSTER COUNTS", audio_cluster_counts)
	print("MOTION CLUSTER COUNTS", motion_cluster_counts)

	print("NUM TOK", len(motion_tokens))
	for tok in motion_tokens:
		print(tok)

	print("NUM AUDIO SAMP", len(audio_tokens))
	print("AUDIO SAMPLES:")
	for samp in audio_tokens:
		print(samp)

	audio_map = {(sample.filename, sample.index):sample for sample in audio_tokens}
	motion_tokens = [tok for tok in motion_tokens if (tok.filename, tok.index) in audio_map]

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
	print(motion_labels_seq)
	motion_encoded = sklearn.preprocessing.label_binarize(motion_labels_seq, classes=list(range(motion_clusters)))

	transition_prob = transition_probability(audio_tokens, audio_clusters)
	motion_transition_prob = transition_probability(motion_tokens, motion_clusters)
	emission_prob = emission_probability(audio_map, audio_clusters, motion_tokens, motion_clusters)

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
		priors = np.zeros(audio_clusters)
		priors[expected_audio[0].cluster] = 1

		result = viterbi([tok.cluster for tok in recon_motion], transition_prob, emission_prob, priors)

		# recon_motion_encoded = sklearn.preprocessing.label_binarize([tok.cluster for tok in recon_motion], classes=list(range(motion_clusters)))
		# result = hmm.predict(recon_motion_encoded)

		for tok, audio in zip(recon_motion, expected_audio):
			if tok.filename != audio.filename or tok.index != audio.index:
				print("MISMATCHED", tok.filename, tok.index, audio.filename, audio.index)

		errors = 0
		for i, cluster in enumerate(result):
			print('{}\t{}\t{}'.format(recon_motion[i].cluster, expected_audio[i].cluster, cluster))
			if cluster != expected_audio[i].cluster:
				errors += 1

		print("ERROR: ", errors / len(result))

	print('Emissions:')
	print_float_2d(emission_prob)
	print('Transitions')
	print_float_2d(transition_prob)
	print('Motion Transitions')
	print_float_2d(motion_transition_prob)
	# for erow in emission_prob:
	# 	for eprob in erow:
	# 		print("{:0.2f}".format(eprob), end=' ')
	# 	print()

	print('Audio Clusters', audio_cluster_counts)
	print('Motion Clusters', motion_cluster_counts)

	motion_centers = motion_classifier.cluster_centers_
	fig_name = '{}_{}_{}_posmotion_cleaned'.format(CLUSTERING, motion_clusters, label)
	max_draw = 50
	for cluster, center in enumerate(motion_centers):
		if CLUSTERING == 'm' and cluster == len(motion_centers) - 1 and len(motion_centers) > 1:
			cluster = -1
		# if motion_cluster_counts[cluster] < 10:
		# 	continue
		curr_draw = 0
		plt.figure(figsize=(8, 10))
		plt.axis('equal')
		for token in motion_tokens:
			if token.cluster != cluster:
				continue
			if curr_draw >= max_draw:
				break
			# curr_skel = token_to_skel(token.motion_diff)
			curr_skel = token.skeletons[0]
			draw_pose(curr_skel, color='b')
			curr_draw += 1
		# cluster_skel = token_to_skel(center)
		cluster_tokens = [m for m in motion_tokens if m.cluster == cluster]
		cluster_skels = [m.skeletons[0] for m in cluster_tokens]
		if len(cluster_skels) == 0:
			plt.close()
			continue
		cluster_skel = {}
		for joint in JOINTS:
			cluster_skel[joint] = (sum([s[joint][0] for s in cluster_skels])/len(cluster_skels), sum([s[joint][1] for s in cluster_skels])/len(cluster_skels))
		draw_pose(cluster_skel, color='r')

		if not os.path.exists('skeletons/{}'.format(fig_name)):
			os.makedirs('skeletons/{}'.format(fig_name))
		plt.savefig('skeletons/{}/{}_{}count.png'.format(fig_name, cluster, motion_cluster_counts[cluster]))
		# plt.show()
		plt.close()

		if not os.path.exists('skeletons/{}/cluster_{}'.format(fig_name, cluster)):
			os.makedirs('skeletons/{}/cluster_{}'.format(fig_name, cluster))

		for i, tok in enumerate(cluster_tokens):
			fig, ax = plt.subplots(nrows=1, ncols=MOTION_LENGTH // 2, sharey=True, figsize=(18, 8))
			fig.subplots_adjust(wspace=0)
			for pos_ind, col in enumerate(ax):
				# col.set_xlim(-1.5, 1.5)
				# col.get_xaxis().set_visible(False)
				col.get_yaxis().set_visible(False)
				skel = tok.skeletons[pos_ind * 2]
				draw_pose(skel, subplt=col)
				if pos_ind == MOTION_LENGTH // 4:
					col.set_title('cluster_{}_motion_{}_{}_{}'.format(cluster, tok.filename, tok.person, tok.index))	
			plt.savefig('skeletons/{}/cluster_{}/motion_{}_{}_{}.png'.format(fig_name, cluster, tok.filename, tok.person, tok.index))
			# plt.show()
			plt.close()



if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='generate tokens')
	# parser.add_argument('-p', '--pickle-data')
	# print(parser.parse_args())
	start_time = time.time()
	main(pickle_data=True, label=LABEL, audio_clusters=N_MUSIC_CLUSTERS, motion_clusters=N_MOTION_CLUSTERS)
	print("This took {:4.2f}s".format(time.time()-start_time))
