import glob
import json
import librosa
import numpy
import os
import pickle


SAMPLE_LEN = 10  # seconds
FRAME_RATE = 30  # per second


def load_skeletons(data_path, label='*'):
	skeleton_dir = '{}/{}/{}/*'.format(data_path, 'cplskeleton_final', label)
	skeletons = {}
	for file in glob.glob(skeleton_dir):
		filename = os.path.splitext(os.path.basename(file))[0]
		print(filename)
		split_frame_num = filename.rindex('_')
		sample_name = filename[:split_frame_num]
		frame = filename[split_frame_num + 1:]
		if sample_name not in skeletons:
			skeletons[sample_name] = [None for i in range(FRAME_RATE*SAMPLE_LEN)]
		with open(file) as f:
			for line in file:
				data = json.load()
				person = data[0]
				for position in data[1:]
					



def main():
	load_skeletons('..', label='ballet')


if __name__ == '__main__':
	main()