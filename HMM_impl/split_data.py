import glob
import os
import shutil


SKEL_DIR = '../skeletons_cleaned_experiment'
AUDIO_DIR = '../audio'

TRAIN_SKEL_DIR = '../skeletons_train_exp'
TEST_SKEL_DIR = '../skeletons_test_exp'
TRAIN_AUDIO_DIR = '../audio_train_exp'
TEST_AUDIO_DIR = '../audio_test_exp'

ALL_DIRS = [TRAIN_AUDIO_DIR, TEST_AUDIO_DIR, TRAIN_SKEL_DIR, TEST_SKEL_DIR]

# train_ratio out of 10, so 2 means 20% data is training
def split_train_test_data(train_ratio=2):
	for folder in ALL_DIRS:
		if not os.path.exists(folder):
			os.mkdir(folder)

	for genre_dir in glob.glob('{}/*'.format(AUDIO_DIR)):
		genre = os.path.splitext(os.path.basename(genre_dir))[0]
		for folder in ALL_DIRS:
			if not os.path.exists('{}/{}'.format(folder, genre)):
				os.mkdir('{}/{}'.format(folder, genre))

		for i, file in enumerate(glob.glob('{}/*'.format(genre_dir))):
			filename = os.path.splitext(os.path.basename(file))[0]
			skel_file = '{}/{}/{}.txt'.format(SKEL_DIR, genre, filename)
			if not os.path.exists(skel_file):
				print('MISSING FILE', skel_file)
				continue
			if i % 10 < train_ratio:
				shutil.copy(file, '{}/{}'.format(TRAIN_AUDIO_DIR, genre))
				shutil.copy(skel_file, '{}/{}'.format(TRAIN_SKEL_DIR, genre))
			else:
				shutil.copy(file, '{}/{}'.format(TEST_AUDIO_DIR, genre))
				shutil.copy(skel_file, '{}/{}'.format(TEST_SKEL_DIR, genre))



split_train_test_data()