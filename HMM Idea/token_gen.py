import numpy as np
import pickle
import glob
import librosa
import os
import time
import sklearn
from sklearn.cluster import KMeans

SAMPLE_RATE = 48000

class AudioSample:
	def __init__(self, filename, index, raw, features, cluster=None):
		self.filename = filename
		self.index = index
		self.raw = raw
		self.features = features
		self.cluster = cluster


# returns a list of (source file, index, sample, features) tuples
def load_audio(data_path, label='*', sample_time=.25, pickle_read=True):
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
	transitions = np.zeros((n_clusters, n_clusters))
	totals = np.zeros(n_clusters)
	for prev, curr in zip(audio_samples[:-1], audio_samples[1:]):
		if prev.filename != curr.filename:
			continue
		transitions[prev.cluster][curr.cluster] += 1
		totals[prev.cluster] += 1
	print(transitions)
	print(totals)
	for i, (tr, to) in enumerate(zip(transitions, totals)):
		print(tr[i], to, tr[i]/to)


if __name__ == '__main__':
	start_time = time.time()
	main(pickle_data=True, label='ballet', n_clusters=48)
	print("This took {:4.2f}s".format(time.time()-start_time))