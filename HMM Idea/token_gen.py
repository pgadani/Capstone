import glob
import json
import librosa
import numpy as np
import os
import pickle
import time

SAMPLE_RATE = 48000
SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second
NUM_SKELETON_POS = SAMPLE_LEN * FRAME_RATE


def load_skeletons(data_path, label='*', use_pickle=True):
	if use_pickle:
		pickle_file = 'pickles/skeleton_{}.pkl'.format(label)
		if os.path.exists(pickle_file):
			return pickle.load(open(pickle_file, 'rb'))
	skeleton_dir = '{}/{}/{}/*'.format(data_path, 'cplskeleton_final', label)
	skeletons = {}
	for file in glob.glob(skeleton_dir):
		filename = os.path.splitext(os.path.basename(file))[0]
		print(filename)
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
	if use_pickle:
		pickle_file = 'pickles/skeleton_{}.pkl'.format(label)
		with open(pickle_file, 'wb+') as f:
			pickle.dump(skeletons, f)
	return skeletons				


def load_audio(data_path, label='*', sample_time=.25, use_pickle=True):
	if use_pickle:
		pickle_file = 'pickles/audio_{}.pkl'.format(label)
		if os.path.exists(pickle_file):
			return pickle.load(open(pickle_file, 'rb'))
	audio_dir = '{}/{}/{}/*'.format(data_path, 'audio', label)
	audio_samples = []
	for audio_file in glob.glob(audio_dir):
		file_duration = librosa.core.get_duration(filename=audio_file, sr=SAMPLE_RATE)
		filename = os.path.splitext(os.path.basename(audio_file))[0]
		print(audio_file)
		for index, offset in enumerate(np.arange(0, file_duration, sample_time)):
			audio_raw, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE, offset=offset, duration=sample_time)
			sample_name = '{}_{}'.format(filename, index)
			audio_chroma = librosa.feature.chroma_stft(audio_raw, sr=SAMPLE_RATE)
			print(index, len(audio_chroma), sample_name)
			audio_samples.append((sample_name, audio_raw, audio_chroma))
	if use_pickle:
		pickle_file = 'pickles/audio_{}.pkl'.format(label)
		with open(pickle_file, 'wb+') as f:
			pickle.dump(audio_samples, f)
	return audio_samples


def main():
	# audio = load_audio('..', label='ballet')
	skeletons = load_skeletons('..', label='ballet')
	for i in range(5):
		print(skeletons.popitem())


if __name__ == '__main__':
	start_time = time.time()
	main()
	print("This took {:4.2f}s".format(time.time()-start_time))