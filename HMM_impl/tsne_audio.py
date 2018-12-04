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
import seaborn as sns
import time

from matplotlib import pyplot as plt
from seqlearn.hmm import MultinomialHMM
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


from token_gen import *


def main(label='*', audio_clusters=25):
	pickle_audio_tok = 'pickles/audio_tokens_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	pickle_audio_feat = 'pickles/audio_features_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	pickle_audio_feat_full = 'pickles/audio_features_full_{}_toksize_{}.pkl'.format(label, TOKEN_SIZE)
	audio_tokens = None
	audio_features = None
	audio_features_full = None

	if os.path.exists(pickle_audio_tok) and os.path.exists(pickle_audio_feat):
		with open(pickle_audio_tok, 'rb') as f:
			audio_tokens = pickle.load(f)
		with open(pickle_audio_feat, 'rb') as f:
			audio_features = pickle.load(f)
		with open(pickle_audio_feat_full, 'rb') as f:
			audio_features_full = pickle.load(f)
	else:
		print('Please generate tokens first with token_gen')
		return

	audio_classifier, audio_labels, audio_clusters = cluster_audio(audio_tokens, audio_features, audio_clusters)

	audio_cluster_counts = np.zeros(audio_clusters)
	for token in audio_tokens:
		audio_cluster_counts[token.cluster] += 1
	print("AUDIO CLUSTER COUNTS", audio_cluster_counts)

	tok_clusters = np.array([tok.cluster for tok in audio_tokens])
	print(tok_clusters)

	sns.set()
	palette = sns.color_palette("hls", audio_clusters)

	embedded_feat = TSNE().fit_transform(audio_features_full)
	print(embedded_feat.shape)

	# cluster_colors = [palette[cluster] for cluster in [tok.cluster for tok in audio_tokens]]

	# plt.scatter(embedded_feat[:,0], embedded_feat[:,1], cluster_colors)


	for cluster in range(audio_clusters):
		feat = embedded_feat[np.where(tok_clusters == cluster)]
		print(feat.shape)
		plt.scatter(feat[:,0], feat[:,1], c=[palette[cluster]])

	plt.legend([str(i) for i in range(audio_clusters)])
	plt.title('{} {} clusters.png'.format(label, audio_clusters))

	if not os.path.exists('tsne_audio'):
		os.makedirs('tsne_audio')
	plt.savefig('tsne_audio/label_{}_clusters_{}.png'.format(label, audio_clusters))
	plt.show()
	plt.close()
	


if __name__ == '__main__':
	main(label=LABEL, audio_clusters=N_MUSIC_CLUSTERS)