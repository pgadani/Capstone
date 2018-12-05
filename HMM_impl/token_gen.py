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
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn import preprocessing

from hmm import *

import hdbscan

SKEL_DIR = '../skeletons_cleaned'
AUDIO_DIR = '../audio'

LABEL = '*'
CLUSTERING = 'swing' # k for kmeans, m for meanshift, d for dbscan, h for hdbscan

SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second

TOKEN_SIZE = 10 # 20 # frames per token
MOTION_STRIDE = 1 # 4 # default is 1
MOTION_LENGTH = TOKEN_SIZE // MOTION_STRIDE

N_CHROMA = 4 # default 12
N_MFCC = 4 # default 20
N_TEMPO = 3 # don't know default

N_MUSIC_CLUSTERS = 12
N_MOTION_CLUSTERS = 12 # 12

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


def load_skeletons(label='*'):
	skeleton_dir = '{}/{}/*'.format(SKEL_DIR, label)
	skeletons = {}
	for file in glob.glob(skeleton_dir):
		filename = os.path.splitext(os.path.basename(file))[0]
		if filename not in skeletons:
			skeletons[filename] = {}
		skeleton = skeletons[filename]
		with open(file) as f:
			data = json.load(f)
			for person, positions in data.items():
				skeleton[person] = {}
				for frame, joints in positions.items():
					skeleton[person][int(frame)] = joints
	return skeletons


def transform(point, offset, angle, scale):
	new_point = [(p - o) * scale for p, o in zip(point, offset)]
	if angle:
		new_point = [new_point[0] * math.cos(angle) - new_point[1] * math.sin(angle), new_point[1] * math.cos(angle) + new_point[0] * math.sin(angle)]
	return new_point


def generate_motion_tokens(skeletons):
	tokens = []
	all_features = []
	for filename, skel in sorted(skeletons.items()):
		for person, positions in sorted(skel.items()):
			for index in range(0, NUM_SKELETON_POS // TOKEN_SIZE - 1):
				i = index * TOKEN_SIZE
				has_all_positions = True
				for k in range(0, TOKEN_SIZE + 1, MOTION_STRIDE):
					if i+k not in positions:
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
				for k in range(0, TOKEN_SIZE + 1, MOTION_STRIDE):
					curr_skel = {}
					for joint in JOINTS:
						curr_skel[joint] = transform(positions[i + k][joint], offset, angle, scale)
					token_skels.append(curr_skel)
				for s in token_skels:
					for j1 in JOINTS:
						for j2 in JOINTS:
							if j1 == j2:
								continue
							motion_features += [s[j1][i] - s[j2][i] for i in range(2)]
				# MOTION
				for s, n in zip(token_skels[:-1], token_skels[1:]):
					for j1 in JOINTS:
						for j2 in JOINTS:
							motion_features += [s[j1][i] - n[j2][i] for i in range(2)]
				# # DIFF FROM START
				# for s in token_skels[1:]:
				# 	s0 = token_skels[0]
				# 	for j1 in JOINTS:
				# 		for j2 in JOINTS:
				# 				motion_features += [s[j1][i] - s0[j2][i] for i in range(2)]
				all_features.append(motion_features)
				tokens.append(MotionToken(filename, person, index, token_skels))

	# features = preprocessing.normalize(np.array(all_features), norm='l2')

	# mean = np.mean(features, axis=0)
	# std = np.sqrt(np.var(features, axis=0))
	# features = (features - mean[np.newaxis, :]) / std[np.newaxis, :]

	# features = preprocessing.normalize(np.array(all_features), norm='l2', axis=0)

	features = preprocessing.scale(np.array(all_features))

	pca = PCA(n_components=.95)
	features = pca.fit_transform(features)
	return tokens, features


def cluster_motion(tokens, features, n_clusters):
	if CLUSTERING == 'm':
		bw = estimate_bandwidth(features)
		classifier = MeanShift(bandwidth=bw, cluster_all=False).fit(features)
		print(classifier.get_params())
		n_clusters = len(classifier.cluster_centers_)
	elif CLUSTERING == 'd':
		eps = 40 # 35 # 28 # 20
		classifier = DBSCAN(eps=eps, min_samples=20).fit(features)
		n_clusters = len(classifier.core_sample_indices_)
	elif CLUSTERING == 'h':
		classifier = hdbscan.HDBSCAN(prediction_data=True).fit(features)
		n_clusters = max(classifier.labels_) + 1
	else:
		classifier = KMeans(n_clusters=n_clusters).fit(features)

	clusters = classifier.labels_
	for token, label in zip(tokens, clusters):
		token.cluster = label
	return classifier, clusters, n_clusters


def load_audio(label='*', sample_time=None):
	if not sample_time:
		sample_time = TOKEN_SIZE / FRAME_RATE

	# aim for sample time between 20 - 40 ms
	# div_h = sample_time / 0.02
	div_l = sample_time / 0.04
	div = int(div_l) + 1 # or int(div_h)
	audio_dir = '{}/{}/*'.format(AUDIO_DIR, label)
	audio_tokens = []
	all_features = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		frame_length = int(sample_time * SAMPLE_RATE / div)
		audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
		features = []
		print(audio_file, frame_length)
		audio_chroma = librosa.feature.chroma_stft(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length, n_chroma=N_CHROMA)
		mfcc = librosa.feature.mfcc(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length, n_mfcc=N_MFCC)
		diff_width = mfcc.shape[-1]
		if diff_width % 2 == 0:
			diff_width -= 1
		if diff_width < 3:
			print('ERROR: MFCC values too small for diff')
		mfcc_diff = librosa.feature.delta(mfcc, width=diff_width)
		tempo = librosa.feature.tempogram(audio_raw, sr=SAMPLE_RATE, hop_length=frame_length, win_length=N_TEMPO)
		onset = librosa.onset.onset_strength(audio_raw, sr=SAMPLE_RATE, n_fft=frame_length, hop_length=frame_length)
		onset = np.reshape(onset, (1, len(onset)))

		print(audio_file, mfcc.shape, audio_chroma.shape, tempo.shape, mfcc_diff.shape)
		features = np.vstack((audio_chroma, mfcc, mfcc_diff, tempo, onset)).T
		# print(audio_file, features.shape)
		num_tok = int(FRAME_RATE * SAMPLE_LEN // TOKEN_SIZE)
		for index in range(num_tok - 1):
			feat = np.mean(features[index*div:(index+1)*div, :], axis=0)
			if np.any(np.isnan(feat)):
				print(audio_file, index)
				print(feat)
				continue
			audio_tokens.append(AudioToken(filename, index))
			all_features.append(feat.T)

	full_audio_features = preprocessing.scale(np.array(all_features))
	pca = PCA(n_components=.95)
	pca_audio_features = pca.fit_transform(full_audio_features)
	print('Full audio', full_audio_features.shape, 'PCA audio', pca_audio_features.shape)
	print(np.where(np.isnan(full_audio_features)))
	return audio_tokens, pca_audio_features, full_audio_features


def cluster_audio(audio_tokens, audio_features, n_clusters):
	# print(audio_features.shape)

	# classifier = MeanShift().fit(audio_features)
	# n_clusters = len(classifier.cluster_centers_)
	classifier = KMeans(n_clusters=n_clusters).fit(audio_features)
	clusters = classifier.labels_
	sequences = []
	for tok, cluster in zip(audio_tokens, clusters):
		tok.cluster = cluster
	return classifier, clusters, n_clusters


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

	# MOTION
	pickle_motion_tok = 'pickles/motion_tokens_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_feat = 'pickles/motion_features_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	motion_tokens = None
	motion_features = None
	if pickle_data:
		if os.path.exists(pickle_motion_tok) and os.path.exists(pickle_motion_feat):
			with open(pickle_motion_tok, 'rb') as f:
				motion_tokens = pickle.load(f)
			with open(pickle_motion_feat, 'rb') as f:
				motion_features = pickle.load(f)

	if motion_tokens is None or motion_features is None: # skeletons is None or
		skeletons = load_skeletons(label=label)
		motion_tokens, motion_features = generate_motion_tokens(skeletons)
		if pickle_data:
			with open(pickle_motion_tok, 'wb+') as f:
				pickle.dump(motion_tokens, f)
			with open(pickle_motion_feat, 'wb+') as f:
				pickle.dump(motion_features, f)

	motion_classifier, motion_labels, motion_clusters = cluster_motion(motion_tokens, motion_features, motion_clusters)
	print(motion_clusters)
	motion_cluster_counts = np.zeros(motion_clusters)
	print('TOKENS')
	for token in motion_tokens:
		if token.cluster != -1:
			motion_cluster_counts[token.cluster] += 1
		print(token)

	print('{} TOKENS WITH {} OUTLIERS:'.format(len(motion_tokens), len([m for m in motion_tokens if m.cluster == -1])))
	print('Motion Clusters', [c for c in motion_cluster_counts if c > 0])


	# AUDIO
	pickle_audio_tok = 'pickles/audio_tokens_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	pickle_audio_feat = 'pickles/audio_features_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	pickle_audio_feat_full = 'pickles/audio_features_full_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	audio_tokens = None
	audio_features = None
	audio_features_full = None
	if pickle_data:
		if os.path.exists(pickle_audio_tok) and os.path.exists(pickle_audio_feat):
			with open(pickle_audio_tok, 'rb') as f:
				audio_tokens = pickle.load(f)
			with open(pickle_audio_feat, 'rb') as f:
				audio_features = pickle.load(f)
			with open(pickle_audio_feat_full, 'rb') as f:
				audio_features_full = pickle.load(f)

	if audio_tokens is None or audio_features is None or audio_features_full is None:
		audio_tokens, audio_features, audio_features_full = load_audio(label=label)
	if pickle_data:
		with open(pickle_audio_tok, 'wb+') as f:
			pickle.dump(audio_tokens, f)
		with open(pickle_audio_feat, 'wb+') as f:
			pickle.dump(audio_features, f)
		with open(pickle_audio_feat_full, 'wb+') as f:
			pickle.dump(audio_features_full, f)

	audio_classifier, audio_labels, audio_clusters = cluster_audio(audio_tokens, audio_features, audio_clusters)
	audio_cluster_counts = np.zeros(audio_clusters)
	for token in audio_tokens:
		audio_cluster_counts[token.cluster] += 1
	print('Audio Clusters', audio_cluster_counts)

	# audio_map = {(tok.filename, tok.index):tok for tok in audio_tokens}
	# transition_prob = transition_probability(audio_tokens, audio_clusters)
	# emission_prob = emission_probability(audio_map, audio_clusters, motion_tokens, motion_clusters)
	# motion_transition_prob = transition_probability(motion_tokens, motion_clusters)

	# print('Emissions:')
	# print_float_2d(emission_prob)
	# print('Transitions')
	# print_float_2d(transition_prob)
	# print('Motion Transitions')
	# print_float_2d(motion_transition_prob)



	fig_name = '{}_{}_{}_toksize_{}_stride_{}_posmotion'.format(CLUSTERING, motion_clusters, label, TOKEN_SIZE, MOTION_STRIDE)
	max_draw = 50
	cluster_centers = {}
	for cluster in range(motion_clusters):
		cluster_tokens = [m for m in motion_tokens if m.cluster == cluster]
		cluster_skels = [m.skeletons[0] for m in cluster_tokens]
		if len(cluster_skels) == 0:
			continue
		if CLUSTERING == 'm' and cluster == len(motion_centers) - 1 and len(motion_centers) > 1:
			cluster = -1
		# if motion_cluster_counts[cluster] < 10:
		# 	continue
		files = {tok.filename for tok in cluster_tokens}
		curr_draw = 0
		plt.figure(figsize=(8, 10))
		plt.axis('equal')
		for i, skel in enumerate(cluster_skels):
			if i >= max_draw:
				break
			draw_pose(skel, color='b')

		cluster_skel = {}
		for joint in JOINTS:
			cluster_skel[joint] = (sum([s[joint][0] for s in cluster_skels])/len(cluster_skels), sum([s[joint][1] for s in cluster_skels])/len(cluster_skels))
		draw_pose(cluster_skel, color='r')
		cluster_centers[cluster] = cluster_skel

		if not os.path.exists('skeletons/{}'.format(fig_name)):
			os.makedirs('skeletons/{}'.format(fig_name))
		plt.savefig('skeletons/{}/cluster_{}_files_{}_tokens_{}count.png'.format(fig_name, cluster, len(files), len(cluster_tokens)))
		# plt.show()
		plt.close()

	# cluster_centers_file = 'skeletons/cluster_centers_{}.txt'.format(fig_name)
	# with open(cluster_centers_file, 'w+') as f:
	# 	json.dump(cluster_centers, f)


	outfile = 'motion_tokens_{}.txt'.format(fig_name)
	with open(outfile, 'w+') as f:
		prev_token = None
		for token in motion_tokens:
			skel_diff = -1
			if token.cluster != -1:
				skel_center = cluster_centers[token.cluster]
				diff = [token.skeletons[0][joint][i] - skel_center[joint][i] for joint in JOINTS for i in range(2)]
				skel_diff = np.linalg.norm(diff)
			# if prev_token is not None and prev_token.cluster != token.cluster:
			if prev_token is not None and (prev_token.filename != token.filename or prev_token.person != token.person or prev_token.index + 1 != token.index):
				print('', file=f)
			print('Filename: {} \t Person: {} \t Index: {} \t Cluster: {} \t Initial distance: {}'.format(token.filename, token.person, token.index, token.cluster, skel_diff), file=f)
			prev_token = token

	outfile = 'motion_tokens_grouped_clusters_{}.txt'.format(fig_name)
	with open(outfile, 'w+') as f:
		prev_token = None
		for token in sorted(motion_tokens, key=lambda tok: tok.cluster):
			skel_diff = -1
			if token.cluster != -1:
				skel_center = cluster_centers[token.cluster]
				diff = [token.skeletons[0][joint][i] - skel_center[joint][i] for joint in JOINTS for i in range(2)]
				skel_diff = np.linalg.norm(diff)
			# if prev_token is not None and prev_token.cluster != token.cluster:
			if prev_token is not None and prev_token.cluster != token.cluster:
				print('', file=f)
			print('Filename: {} \t Person: {} \t Index: {} \t Cluster: {} \t Initial distance: {}'.format(token.filename, token.person, token.index, token.cluster, skel_diff), file=f)
			prev_token = token

	# for cluster in range(motion_clusters):
	# 	if CLUSTERING == 'm' and cluster == len(motion_centers) - 1 and len(motion_centers) > 1:
	# 		cluster = -1
	# 	cluster_tokens = [m for m in motion_tokens if m.cluster == cluster]
	# 	cluster_skels = [m.skeletons[0] for m in cluster_tokens]
	# 	if len(cluster_skels) == 0:
	# 		continue
	# 	if not os.path.exists('skeletons/{}/cluster_{}'.format(fig_name, cluster)):
	# 		os.makedirs('skeletons/{}/cluster_{}'.format(fig_name, cluster))


	# 	for i, tok in enumerate(cluster_tokens):
	# 		if i % 3 == 0:
	# 			fig, ax = plt.subplots(nrows=1, ncols=MOTION_LENGTH // 2, sharey=True, figsize=(18, 8))
	# 			fig.subplots_adjust(wspace=0)
	# 			for pos_ind, col in enumerate(ax):
	# 				# col.set_xlim(-1.5, 1.5)
	# 				# col.get_xaxis().set_visible(False)
	# 				col.get_yaxis().set_visible(False)
	# 				skel = tok.skeletons[pos_ind * 2]
	# 				draw_pose(skel, subplt=col)
	# 				if pos_ind == MOTION_LENGTH // 4:
	# 					col.set_title('cluster_{}_motion_{}_{}_{}'.format(cluster, tok.filename, tok.person, tok.index))
	# 			plt.savefig('skeletons/{}/cluster_{}/motion_{}_{}_{}.png'.format(fig_name, cluster, tok.filename, tok.person, tok.index))
	# 			# plt.show()
	# 			plt.close()


if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label=LABEL, audio_clusters=N_MUSIC_CLUSTERS, motion_clusters=N_MOTION_CLUSTERS)
	print("This took {:4.2f}s".format(time.time()-start_time))
