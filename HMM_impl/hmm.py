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

from token_gen import *


def print_float_2d(arr, dec=2):
	format_str = '{{:0.{}f}}'.format(dec)
	for row in arr:
		for val in row:
			print(format_str.format(val), end=' ')
		print()


# Generates emission probability matrix
# Probability of observing audio from state of motion
def emission_probability(audio_tok_map, audio_clusters, motion_tokens, motion_clusters):
	transitions = np.zeros((audio_clusters, motion_clusters))
	for motion_token in motion_tokens:
		if motion_token.cluster == -1:
			continue
		if (motion_token.filename, motion_token.index) in audio_tok_map:
			audio_token = audio_tok_map[(motion_token.filename, motion_token.index)]
			transitions[audio_token.cluster, motion_token.cluster] += 1
	# row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = normalize(transitions, norm='l1')
	return probabilities


def transition_probability(audio_tokens, audio_clusters):
	transitions = np.zeros((audio_clusters, audio_clusters))
	for prev, curr in zip(audio_tokens[:-1], audio_tokens[1:]):
		if prev.filename != curr.filename or prev.cluster == -1 or curr.cluster == -1:
			continue
		transitions[prev.cluster][curr.cluster] += 1
	probabilities = normalize(transitions, norm='l1')
	return probabilities


def main(pickle_data=True, label='*', audio_clusters=25, motion_clusters=25):
	pickle_motion_tok = 'pickles/motion_tokens_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_feat = 'pickles/motion_features_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	motion_tokens = None
	motion_features = None

	pickle_audio_tok = 'pickles/audio_tokens_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	pickle_audio_feat = 'pickles/audio_features_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	audio_tokens = None
	audio_features = None

	if os.path.exists(pickle_motion_tok) and os.path.exists(pickle_motion_feat):
		with open(pickle_motion_tok, 'rb') as f:
			motion_tokens = pickle.load(f)
		with open(pickle_motion_feat, 'rb') as f:
			motion_features = pickle.load(f)
	else:
		print('Please generate tokens first with token_gen')
		return

	if os.path.exists(pickle_audio_tok): # and os.path.exists(pickle_audio_feat):
		with open(pickle_audio_tok, 'rb') as f:
			audio_tokens = pickle.load(f)
		with open(pickle_audio_feat, 'rb') as f:
			audio_features = pickle.load(f)
	else:
		print('Please generate tokens first with token_gen')
		return


	audio_classifier, audio_labels, audio_clusters = cluster_audio(audio_tokens, audio_features, audio_clusters)
	motion_classifier, motion_labels, motion_clusters = cluster_motion(motion_tokens, motion_features, motion_clusters)

	audio_cluster_counts = np.zeros(audio_clusters)
	for token in audio_tokens:
		audio_cluster_counts[token.cluster] += 1

	motion_cluster_counts = np.zeros(motion_clusters)
	for token in motion_tokens:
		motion_cluster_counts[token.cluster] += 1

	print("AUDIO CLUSTER COUNTS", audio_cluster_counts)
	print("MOTION CLUSTER COUNTS", motion_cluster_counts)

	print("NUM TOK", len(motion_tokens))
	for tok in motion_tokens:
		print(tok)

	print("NUM AUDIO SAMP", len(audio_tokens))
	print("AUDIO SAMPLES:")
	for tok in audio_tokens:
		print(tok)

	audio_cluster_centers = audio_classifier.cluster_centers_

	audio_map = {(tok.filename, tok.index):tok for tok in audio_tokens}
	motion_tokens = [tok for tok in motion_tokens if (tok.filename, tok.index) in audio_map]

	# motion_labels_seq = [tok.cluster for tok in motion_tokens]
	# audio_labels_seq = np.array([audio_map[(tok.filename, tok.index)].cluster for tok in motion_tokens])

	# sequence_lens = []
	# curr_len = 1
	# for prev, curr in zip(motion_tokens[:-1], motion_tokens[1:]):
	# 	if curr.filename != prev.filename or curr.person != prev.person or curr.index - prev.index != 1:
	# 		sequence_lens.append(curr_len)
	# 		curr_len = 0
	# sequence_lens.append(curr_len)

	# Make an actual label binarizer object
	# print(motion_labels_seq)
	# motion_encoded = sklearn.preprocessing.label_binarize(motion_labels_seq, classes=list(range(motion_clusters)))

	transition_prob = transition_probability(audio_tokens, audio_clusters)
	motion_transition_prob = transition_probability(motion_tokens, motion_clusters)
	emission_prob = emission_probability(audio_map, audio_clusters, motion_tokens, motion_clusters)

	# print('MOTION ONE-HOT SHAPE', motion_encoded.shape)
	# print('AUDIO LABELS SHAPE', audio_labels_seq.shape)
	# print('SEQUENCE LENS SUM', sum(sequence_lens))

	# hmm = MultinomialHMM()
	# hmm.fit(motion_encoded, audio_labels_seq, sequence_lens)

	all_samples = {(tok.filename, tok.index, tok.person) for tok in motion_tokens}
	recon_samples = []

	for samp in all_samples:
		audio_samp = (samp[0], samp[1])
		if audio_samp in audio_map:
			recon_samples.append(samp)
			if len(recon_samples) >= 80:
				break

	fig_name = '{}_{}_{}_toksize_{}_stride_{}_{}'.format(CLUSTERING, motion_clusters, label, TOKEN_SIZE, MOTION_STRIDE, RUN_NAME)
	file = 'viterbi_out_{}.txt'.format(fig_name)
	with open(file, 'w+') as f:
		print('Motion cluster \tTrue audio cluster \tPredicted audio cluster \tDistance to true center \tDistance to predicted center \tDistance between true and predicted', file=f)
		for samp in recon_samples:
			print(samp[0], samp[1], file=f)
			recon_motion = list(filter(lambda tok: tok.filename == samp[0] and tok.person == samp[2], motion_tokens))
			recon_motion.sort(key=lambda tok: tok.index)

			expected_audio = [audio_map[(tok.filename, tok.index)] for tok in recon_motion]
			priors = np.zeros(audio_clusters)
			priors[expected_audio[0].cluster] = 1

			if CLUSTERING == 'h':
				features = [motion_features[tok.index, :] for tok in recon_motion]
				motion_labels, _strengths = hdbscan.approximate_predict(motion_classifier, features)
			else:
				motion_labels = [tok.cluster for tok in recon_motion]

			result = viterbi(motion_labels, transition_prob, emission_prob, priors)

			# recon_motion_encoded = sklearn.preprocessing.label_binarize([tok.cluster for tok in recon_motion], classes=list(range(motion_clusters)))
			# result = hmm.predict(recon_motion_encoded)

			for tok, audio in zip(recon_motion, expected_audio):
				if tok.filename != audio.filename or tok.index != audio.index:
					print("MISMATCHED", tok.filename, tok.index, audio.filename, audio.index, file=f)

			errors = 0
			for i, (cluster, tok) in enumerate(zip(result, expected_audio)):
				expected_center = audio_cluster_centers[expected_audio[i].cluster, :]
				predicted_center = audio_cluster_centers[cluster, :]

				feat_index = audio_tokens.index(tok)
				actual_feat = audio_features[feat_index, :]

				print('{}\t{}\t{}\t{}\t{}\t{}'.format(recon_motion[i].cluster, expected_audio[i].cluster, cluster, np.linalg.norm(actual_feat - expected_center), np.linalg.norm(actual_feat - predicted_center), np.linalg.norm(expected_center - predicted_center)), file=f)
				if cluster != expected_audio[i].cluster:
					errors += 1

			print("ERROR: ", errors / len(result), file=f)

	print('Emissions:')
	print_float_2d(emission_prob)
	print('Transitions')
	print_float_2d(transition_prob)
	print('Motion Transitions')
	print_float_2d(motion_transition_prob)




if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label=LABEL, audio_clusters=N_MUSIC_CLUSTERS, motion_clusters=N_MOTION_CLUSTERS)
	print("This took {:4.2f}s".format(time.time()-start_time))
