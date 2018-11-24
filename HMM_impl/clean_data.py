import glob
import json
import os

DATA_DIR = '../cplskeleton_final'
DEST_DIR = '../skeletons_cleaned'

NUM_FRAMES = 300  # 300 frames for each video

# dictionary hashed genre -> sample -> frame -> person -> joints

def load_skeletons(dir):
	skeleton_genres = {}
	for genre_dir in glob.glob('{}/*'.format(dir)):
		genre = os.path.splitext(os.path.basename(genre_dir))[0]

		skeletons = skeleton_genres[genre] = {}
		for file in glob.glob('{}/*'.format(genre_dir)):
			filename = os.path.splitext(os.path.basename(file))[0]
			split_frame_num = filename.rindex('_')
			sample_name = filename[:split_frame_num]
			frame = int(filename[split_frame_num + 1:]) - 1
			if sample_name not in skeletons:
				skeletons[sample_name] = [{} for i in range(NUM_FRAMES)]
			skeleton = skeletons[sample_name][frame]
			with open(file) as f:
				for line in f:
					data = json.loads(line)
					person = data[0]
					if person not in skeletons[sample_name]:
						skeleton[person] = {}
					for position in data[1:]:
						joint, x, y = position
						skeleton[person][joint] = (x, y)
	return skeleton_genres


def center(p1, p2):
	return[(p1[i] + p2[i])/2 for i in range(2)]


# distance metric used to match skeletons
def match_dist(s1, s2):
	c1 = center(s1['Lhip'], s1['Rhip'])
	c2 = center(s2['Lhip'], s2['Rhip'])
	return sum([(c1[i] - c2[i])**2 for i in range(2)])**.5


'''given two frames of skeletons, match skeletons in frame 1 to those in frame 2
does a naive greedy match
if there are more skeletons in prev than in curr,
	it switches to do the reverse match and swaps the results
return a list of tuples corresponding to (previous, current) skeleton
'''
def find_matches(prev, curr):
	matches = []
	switched = False
	used = set()
	max_prev = 0 if len(prev) == 0 else max(prev.keys()) + 1
	if len(prev) > len(curr):
		switched = True
		prev, curr = curr, prev
	for p, ps in prev.items():
		min_dist = 10000 # larger than diagonal distance for 1024 x 768
		min_person = None
		for c, cs in curr.items():
			if c in used:
				continue
			dist = match_dist(ps, cs)
			if dist < min_dist:
				min_dist = dist
				min_person = c
		# this should not be triggered
		# unless a potential match is worse than min_dist
		if min_person is None:
			continue
		matches.append((p, min_person))
		used.add(min_person)
	if switched:
		return [(m[1], m[0]) for m in matches]
	for c in curr.keys():
		if c not in used:
			matches.append((max_prev, c))
			max_prev += 1
	return matches


def clean_skeletons(all_skeletons):
	new_skeletons = {}
	for genre, genre_skeletons in all_skeletons.items():
		new_skeletons[genre] = {}
		for file, skeleton_file in genre_skeletons.items():
			new_file = new_skeletons[genre][file] = [{} for i in range(NUM_FRAMES)]
			prev_people = new_file[0] = {i: v for i, v in enumerate(skeleton_file[0].values())}
			# print(file)
			for frame, skeleton_frame in enumerate(skeleton_file):
				if frame == 0:
					continue
				matches = find_matches(prev_people, skeleton_frame)
				for pm, cm in matches:
					new_file[frame][pm] = skeleton_frame[cm]
				prev_people = new_file[frame]
	return new_skeletons


def save_skeletons(save_dir, all_skeletons):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for genre, genre_skeletons in all_skeletons.items():
		genre_dir = '{}/{}'.format(save_dir, genre)
		if not os.path.exists(genre_dir):
			os.makedirs(genre_dir)
		for file, skeleton_file in genre_skeletons.items():
			filename = '{}/{}.txt'.format(genre_dir, file)
			print(filename)
			with open(filename, 'w+') as f:
				json.dump(skeleton_file, f)


def main():
	skeletons = load_skeletons(DATA_DIR)
	# print(skeletons.keys())
	# print(skeletons['ballet'].keys())
	# print(skeletons['ballet']['HEM5cG_43vo_500'][0].keys())
	# print(skeletons['ballet']['HEM5cG_43vo_500'][0]['person0'])
	cleaned = clean_skeletons(skeletons)
	save_skeletons(DEST_DIR, cleaned)


if __name__ == '__main__':
	main()
