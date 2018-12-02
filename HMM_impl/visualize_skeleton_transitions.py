import glob
import json
import math
import os
import numpy as np
from matplotlib import pyplot as plt

from token_gen import load_skeletons


NUM_FRAMES = 300  # 300 frames for each video
SKEL_DIR = '../skeletons_cleaned_new'
SAVE_DIR = '../diffs_new'

JOINTS = ["head", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank"]

hist_bins = 100


def transform(point, offset, angle, scale):
	new_point = [(p - o) * scale for p, o in zip(point, offset)]
	if angle:
		new_point = [new_point[0] * math.cos(angle) - new_point[1] * math.sin(angle), new_point[1] * math.cos(angle) + new_point[0] * math.sin(angle)]
	return new_point


def skel_diff(s1, s2):
	# use center of hips as origin
	offset = [(s1['Lhip'][j] + s1['Rhip'][j])/2 for j in range(2)]
	# scale based on neck being one unit away
	diff = [s1['neck'][j] - offset[j] for j in range(2)]
	# not messing with angle for now
	# angle = math.atan2(*diff)
	angle = 0
	if diff[0]**2 + diff[1]**2 == 0:
		return None
	scale = 1/math.sqrt(diff[0]**2 + diff[1]**2)
	total_diff = 0
	for j in JOINTS:
		n1 = transform(s1[j], offset, angle, scale)
		n2 = transform(s2[j], offset, angle, scale)
		total_diff += sum([(n1[i] - n2[i])**2 for i in range(2)])
	return total_diff**.5


def visualize_all_diffs(skeletons):
	flat = []
	for filename, skel_file in skeletons.items():
		for person, positions in skel_file.items():
			for f1 in sorted(list(positions.keys())):
				if f1 + 1 not in positions:
					continue
				p1 = positions[f1]
				p2 = positions[f1+1]
				if 'head' not in p1 or 'head' not in p2:
					continue
				diff = skel_diff(p1, p2)
				if diff is not None:
					flat.append(diff)
	print(len(flat))
	hist, borders = np.histogram(flat, bins=hist_bins)
	print(hist[:10])
	plt.figure()
	plt.hist(flat, bins=hist_bins)
	plt.show()
	flat_filtered = [f for f in flat if f < borders[1]]
	hist, borders = np.histogram(flat_filtered, bins=20)
	print(hist[:10])
	plt.figure()
	plt.hist(flat_filtered, bins=20)
	plt.savefig('filtered_diff_histogram.png')
	plt.show()


def plot_per_file(skeletons):
	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	for filename, skel_file in skeletons.items():
		plt.figure(figsize=(9.6, 4.8))
		plt.ylim(0, 5)
		for person, positions in skel_file.items():
			diffs = np.zeros(NUM_FRAMES-1)
			for f1 in sorted(list(positions.keys())):
				if f1 + 1 not in positions:
					continue
				p1 = positions[f1]
				p2 = positions[f1+1]
				if 'head' not in p1 or 'head' not in p2:
					continue
				diff = skel_diff(p1, p2)
				if diff is not None:
					diffs[f1] = diff
			plt.plot(diffs)
		plt.savefig('{}/diffs_{}.png'.format(SAVE_DIR, filename))
		plt.close()


def main():
	skeletons = load_skeletons()
	print('Loaded Skeletons')
	# visualize_all_diffs(skeletons)
	plot_per_file(skeletons)


if __name__ == '__main__':
	main()
