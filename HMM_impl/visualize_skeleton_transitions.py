import glob
import json
import math
import os
import numpy as np
from matplotlib import pyplot as plt


NUM_FRAMES = 300  # 300 frames for each video
SKEL_DIR = '../skeletons_cleaned_exp'

JOINTS = ["head", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank"]

hist_bins = 100


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
						skeleton[person] = [{} for i in range(NUM_FRAMES)]
					for joint, pos in joints.items():
						skeleton[person][frame][joint] = pos
			for person in list(skeleton.keys()):
				if len([frame for frame in skeleton[person] if frame]) < 0.1 * NUM_FRAMES:
					del skeleton[person]
	return skeletons


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
			for p1, p2 in zip(positions[:-1], positions[1:]):
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
	for filename, skel_file in skeletons.items():
		plt.figure(figsize=(9.6, 4.8))
		plt.ylim(0, 5)
		for person, positions in skel_file.items():
			diffs = np.zeros(NUM_FRAMES-1)
			for i, (p1, p2) in enumerate(zip(positions[:-1], positions[1:])):
				if 'head' not in p1 or 'head' not in p2:
					continue
				diff = skel_diff(p1, p2)
				if diff is not None:
					diffs[i] = diff
			plt.plot(diffs)
		plt.savefig('../diff_skeletons/diffs_{}.png'.format(filename))
		plt.close()


def main():
	skeletons = load_skeletons()
	print('Loaded Skeletons')
	# visualize_all_diffs(skeletons)
	plot_per_file(skeletons)


if __name__ == '__main__':
	main()
