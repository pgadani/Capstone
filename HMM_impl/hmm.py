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
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


def transition_probability(audio_tokens, audio_clusters):
	transitions = np.zeros((audio_clusters, audio_clusters))
	for prev, curr in zip(audio_tokens[:-1], audio_tokens[1:]):
		if prev.filename != curr.filename or prev.cluster == -1 or curr.cluster == -1:
			continue
		transitions[prev.cluster][curr.cluster] += 1
	row_sums = transitions.sum(axis=1, keepdims=True)
	probabilities = transitions / row_sums
	return probabilities


# TODO CLEAN THIS AFTER FINISHING WITH FIXING TOKEN GEN
def main(pickle_data=True, label='*', audio_clusters=25, motion_clusters=25):
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
	for tok in audio_tokens:
		print(tok)

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





if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label=LABEL, audio_clusters=N_MUSIC_CLUSTERS, motion_clusters=N_MOTION_CLUSTERS)
	print("This took {:4.2f}s".format(time.time()-start_time))
