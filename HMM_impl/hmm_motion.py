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
from pomegranate import HiddenMarkovModel, NormalDistribution
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import normalize

from hmm import print_float_2d, transition_probability
from token_gen import *


# COLORS = [[240,163,255], [0,117,220], [153,63,0], [76,0,92], [25,25,25], 
# 		  [0,92,49], [43,206,72], [255,204,153], [128,128,128], [148,255,181],
# 		  [143,124,0], [157,204,0], [194,0,136], [0,51,128], [255,164,5],
# 		  [255,168,187], [66,102,0], [255,0,16], [94,241,242], [0,153,143],
# 		  [224,255,102], [116,10,255], [153,0,0], [255,255,128], [255,255,0], [255,80,5]]

COLORS = [[0.9375, 0.63671875, 0.99609375], [0.0, 0.45703125, 0.859375], [0.59765625, 0.24609375, 0.0], [0.296875, 0.0, 0.359375], [0.09765625, 0.09765625, 0.09765625], [0.0, 0.359375, 0.19140625], [0.16796875, 0.8046875, 0.28125], [0.99609375, 0.796875, 0.59765625], [0.5, 0.5, 0.5], [0.578125, 0.99609375, 0.70703125], [0.55859375, 0.484375, 0.0], [0.61328125, 0.796875, 0.0], [0.7578125, 0.0, 0.53125], [0.0, 0.19921875, 0.5], [0.99609375, 0.640625, 0.01953125], [0.99609375, 0.65625, 0.73046875], [0.2578125, 0.3984375, 0.0], [0.99609375, 0.0, 0.0625], [0.3671875, 0.94140625, 0.9453125], [0.0, 0.59765625, 0.55859375], [0.875, 0.99609375, 0.3984375], [0.453125, 0.0390625, 0.99609375], [0.59765625, 0.0, 0.0], [0.99609375, 0.99609375, 0.5], [0.99609375, 0.99609375, 0.0], [0.99609375, 0.3125, 0.01953125]]


def main(label='*', motion_clusters=25):

	pickle_model = 'pickles/hmm_model_{}_toksize_{}_stride_{}_states_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE, motion_clusters)

	pickle_motion_tok = 'pickles/motion_tokens_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_feat = 'pickles/motion_features_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_pca = 'pickles/motion_pca_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)
	pickle_motion_scaler = 'pickles/motion_scaler_{}_toksize_{}_stride_{}.pkl'.format(label, TOKEN_SIZE, MOTION_STRIDE)

	motion_tokens = None
	motion_features = None
	motion_pca = None
	motion_scaler = None

	if os.path.exists(pickle_motion_tok) and os.path.exists(pickle_motion_feat) and os.path.exists(pickle_motion_pca) and os.path.exists(pickle_motion_scaler):
		with open(pickle_motion_tok, 'rb') as f:
			motion_tokens = pickle.load(f)
		with open(pickle_motion_feat, 'rb') as f:
			motion_features = pickle.load(f)
		with open(pickle_motion_pca, 'rb') as f:
			motion_pca = pickle.load(f)
		with open(pickle_motion_scaler, 'rb') as f:
			motion_scaler = pickle.load(f)
	else:
		print('Please generate tokens first with token_gen')
		return


	tok_sequences = []
	feat_sequences = []
	curr_tok_seq = [motion_tokens[0]]
	curr_seq = [motion_features[0,:]]
	for prev, curr, feat in zip(motion_tokens[:-1], motion_tokens[1:], motion_features[1:]):
		if curr.filename != prev.filename or curr.person != prev.person or curr.index - prev.index != 1:
			seq = np.array(curr_seq)
			feat_sequences.append(seq)
			curr_seq = []
			tok_sequences.append(curr_tok_seq)
			curr_tok_seq = []
		curr_seq.append(feat)
		curr_tok_seq.append(curr)
	feat_sequences.append(curr_seq)
	tok_sequences.append(curr_tok_seq)



	if os.path.exists(pickle_model):
		with open(pickle_model, 'rb') as f:
			model = pickle.load(f)
	else:
		model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=motion_clusters, X=feat_sequences)
		with open(pickle_model, 'wb+') as f:
			pickle.dump(model, f)

	print('Model transition matrix')
	print(model.dense_transition_matrix())

	state_token_map = {}
	tok_state_map = {}
	for tok_seq, feat_seq in zip(tok_sequences, feat_sequences):
		first_tok = tok_seq[0]
		states = model.predict(feat_seq)
		for tok, state in zip(tok_seq, states):
			if state not in state_token_map:
				state_token_map[state] = []
			state_token_map[state].append(tok)
			tok_state_map[tok] = state


	fig_name = '{}_{}_{}_toksize_{}_stride_{}'.format('hmm', motion_clusters, label, TOKEN_SIZE, MOTION_STRIDE)
	max_draw = 50
	state_centers = {}
	for state in range(motion_clusters):
		if state not in state_token_map:
			print('EMPTY STATE', state)
			continue
		state_tokens = state_token_map[state]
		state_skels = [m.skeletons[0] for m in state_tokens]
		if len(state_skels) == 0:
			continue
		files = {tok.filename for tok in state_tokens}
		curr_draw = 0
		plt.figure(figsize=(8, 10))
		plt.axis('equal')
		random.shuffle(state_skels)
		for i, skel in enumerate(state_skels):
			if i >= max_draw:
				break
			draw_pose(skel, color='b')

		center_skel = {}
		for joint in JOINTS:
			center_skel[joint] = (sum([s[joint][0] for s in state_skels])/len(state_skels), sum([s[joint][1] for s in state_skels])/len(state_skels))
		draw_pose(center_skel, color='r')
		state_centers[state] = center_skel

		if not os.path.exists('skeletons/{}'.format(fig_name)):
			os.makedirs('skeletons/{}'.format(fig_name))
		plt.savefig('skeletons/{}/cluster_{}_files_{}_tokens_{}count.png'.format(fig_name, state, len(files), len(state_tokens)))
		# plt.show()
		plt.close()


	if not os.path.exists('skeleton_dist/{}/'.format(fig_name)):
		os.makedirs('skeleton_dist/{}/'.format(fig_name))

	# Plot the distance to cluster center over time, current cluster, and current skeleton
	for tok_seq in tok_sequences:
		if len(tok_seq) < 50:
			continue
		center_diffs = []
		states = []
		for tok in tok_seq:
			skel_center = state_centers[tok_state_map[tok]]
			diff = [tok.skeletons[0][joint][i] - skel_center[joint][i] for joint in JOINTS for i in range(2)]
			center_diffs.append(np.linalg.norm(diff))
			states.append(tok_state_map[tok])
		scatter_colors = np.array([COLORS[state] for state in states])
		plt.figure(figsize=(15, 8))
		plt.xlim((0, 150))
		plt.ylim((-0.25, 2))
		plt.scatter(list(range(len(tok_seq))), center_diffs, c=scatter_colors, s=0.1)
		for i, (tok, diff, state) in enumerate(zip(tok_seq, center_diffs, states)):
			scale = 3
			# yoff = i%10/20 if (i/10)%2 == 0 else (10-i%10)/20
			draw_pose(tok.skeletons[0], color=COLORS[state], xscale=scale, yscale=scale/75, offset=(i, -diff), draw_head=False)
			draw_pose(state_centers[state], color=COLORS[state], xscale=scale, yscale=scale/75, offset=(i, 0), draw_head=False)
		tok = tok_seq[0]
		plt.title('Distance from States: file {} person {} index {}'.format(tok.filename, tok.person, tok.index))
		plt.savefig('skeleton_dist/{}/skel_dist_{}_{}_{}'.format(fig_name, tok.filename, tok.person, tok.index))
		# plt.show()
		plt.close()


	for run_index in range(20):
		samples, path = model.sample(30, path=True)
		skels = []
		for samp, state in zip(samples, path):
			orig_scaled = motion_pca.inverse_transform(samp)
			orig_feat = motion_scaler.inverse_transform(orig_scaled)

			system_rows = []
			for posi, j1 in enumerate(JOINTS):
				for negi, j2 in enumerate(JOINTS):
					if j1 == j2:
						continue
					rowx = np.zeros(2*len(JOINTS))
					rowx[2*posi] = 1
					rowx[2*negi] = -1
					rowy = np.zeros(2*len(JOINTS))
					rowy[2*posi + 1] = 1
					rowy[2*negi + 1] = -1
					system_rows.append(rowx)
					system_rows.append(rowy)
			system = np.array(system_rows)
			linreg = Ridge(fit_intercept=False).fit(system, orig_feat)
			joint_pos = linreg.coef_
			skel = {joint:(posx, posy) for joint, posx, posy in zip(JOINTS, joint_pos[::2], joint_pos[1::2])}
			
			skels.append(skel)

		plt.figure(figsize=(15, 8))
		plt.axis('equal')
		for i, (skel, state) in enumerate(zip(skels, path)):
			if i==0:
				color = 'k'
			else:
				color = COLORS[int(state.name[1:])]
			draw_pose(skel, offset=(2*i, 0), color=color)
		if not os.path.exists('hmm_motion_gen_{}_STATE_COLS/ridge_seq_{}/'.format(fig_name, run_index)):
			os.makedirs('hmm_motion_gen_{}_STATE_COLS/ridge_seq_{}/'.format(fig_name, run_index))
		plt.savefig('hmm_motion_gen_{}_PRETTY/ridge_seq_{}/gen_ridge_{}'.format(fig_name, run_index, run_index))
		plt.show()
		plt.close()

		# for chunk_index in range(6):
		# 	fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(18, 8))
		# 	fig.subplots_adjust(wspace=0)
		# 	for skel, col in zip(skels[5*chunk_index:5*(chunk_index+1)], ax):
		# 		col.get_yaxis().set_visible(False)
		# 		draw_pose(skel, subplt=col)
		# 	ax[0].get_yaxis().set_visible(True)
		# 	if not os.path.exists('hmm_motion_gen_{}_FIN/ridge_seq_{}/'.format(fig_name, run_index)):
		# 		os.makedirs('hmm_motion_gen_{}_FIN/ridge_seq_{}/'.format(fig_name, run_index))
		# 	plt.savefig('hmm_motion_gen_{}_FIN/ridge_seq_{}/gen_ridge_{}_{}'.format(fig_name, run_index, run_index, chunk_index))
		# 	# plt.show()
		# 	plt.close()


if __name__ == '__main__':
	start_time = time.time()
	main(label=LABEL)
	print("This took {:4.2f}s".format(time.time()-start_time))
