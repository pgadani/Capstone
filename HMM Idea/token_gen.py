import numpy as np
import pickle
import glob
import librosa
import os
import time

SAMPLE_RATE = 48000

def load_audio(data_path, label='*', sample_time=.25, pickle_read=True):
	if pickle_read:
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
	if pickle_read:
		pickle_file = 'pickles/audio_{}.pkl'.format(label)
		pickle.dump(audio_samples, open(pickle_file, 'wb+'))
	return audio_samples

def main():
	audio = load_audio('..', label='ballet')

if __name__ == '__main__':
	start_time = time.time()
	main()
	print("This took {:4.2f}s".format(time.time()-start_time))