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

SKEL_DIR = '../skeletons_cleaned_experiment'
AUDIO_DIR = '../audio_train'

LABEL = '*'
CLUSTERING = 'k' # k for kmeans, m for meanshift, d for dbscan

SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second

TOKEN_SIZE = 10 # frames per token
MOTION_STRIDE = 1 # default is 1
MOTION_LENGTH = TOKEN_SIZE // MOTION_STRIDE

N_CHROMA = 4 # default 12
N_MFCC = 4 # default 20
N_TEMPO = 3 # don't know default

N_MUSIC_CLUSTERS = 12
N_MOTION_CLUSTERS = 32 # 25 # 12

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
	all_features = []
	for filename, skel in sorted(skeletons.items()):
		for person, positions in sorted(skel.items()):
			for index in range(0, NUM_SKELETON_POS // TOKEN_SIZE - 1):
				i = index * TOKEN_SIZE
				has_all_positions = True
				for k in range(0, TOKEN_SIZE + 1, MOTION_STRIDE):
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
				mismatch = False
				for k in range(0, TOKEN_SIZE + 1, MOTION_STRIDE):
					curr_skel = {}
					for joint in JOINTS:
						curr_skel[joint] = transform(positions[i + k][joint], offset, angle, scale)
					token_skels.append(curr_skel)
					if (curr_skel['Lhip'][0] + curr_skel['Rhip'][0]) > 4:
						mismatch = True
						print('possible mismatch', filename, person, index)
						break
				if mismatch:
					fig, ax = plt.subplots(nrows=1, ncols=MOTION_LENGTH, sharey=True, figsize=(18, 8))
					fig.subplots_adjust(wspace=0)
					for pos_ind, (col, skel) in enumerate(zip(ax, token_skels)):
						# col.set_xlim(-1.5, 1.5)
						# col.get_xaxis().set_visible(False)
						col.get_yaxis().set_visible(False)
						draw_pose(skel, subplt=col)
						if pos_ind == MOTION_LENGTH // 4:
							col.set_title('motion_{}_{}_{}'.format(filename, person, index))
					plt.savefig('skeletons/debugging_experiment/motion_{}_{}_{}.png'.format(filename, person, index))
					# plt.show()
					plt.close()

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

	# pca = PCA(n_components=.95)
	# features = pca.fit_transform(features)
	return tokens, features


def cluster_motion(tokens, features, n_clusters):
	if CLUSTERING == 'm':
		bw = estimate_bandwidth(features)
		classifier = MeanShift(bandwidth=bw, cluster_all=False).fit(features)
		print(classifier.get_params())
		n_clusters = len(classifier.cluster_centers_)
	elif CLUSTERING == 'd':
		eps = 28 # 20
		classifier = DBSCAN(eps=eps, min_samples=5).fit(features)
		n_clusters = len(classifier.core_sample_indices_)
	else:
		classifier = KMeans(n_clusters=n_clusters).fit(features)

	clusters = classifier.labels_
	for token, label in zip(tokens, clusters):
		token.cluster = label
	return classifier, clusters, n_clusters


# returns a list of (source file, index, sample, features) tuples
def load_audio(label='*', sample_time=None):
	if not sample_time:
		sample_time = TOKEN_SIZE / FRAME_RATE
	audio_dir = '{}/{}/*'.format(AUDIO_DIR, label)
	audio_tokens = []
	all_features = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		frame_length = int(sample_time * SAMPLE_RATE)
		audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
		features = []
		print(audio_file)
		audio_chroma = librosa.feature.chroma_cqt(audio_raw, sr=SAMPLE_RATE, hop_length=frame_length, n_chroma=N_CHROMA)
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
	# pickle_skeletons = 'pickles/skeletons_{}.pkl'.format(label)
	pickle_motion_tok = 'pickles/motion_tokens_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_feat = 'pickles/motion_features_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	motion_tokens = None
	motion_features = None

	skeletons = load_skeletons(label=label)
	motion_tokens, motion_features = generate_motion_tokens(skeletons)


if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label=LABEL, audio_clusters=N_MUSIC_CLUSTERS, motion_clusters=N_MOTION_CLUSTERS)
	print("This took {:4.2f}s".format(time.time()-start_time))
