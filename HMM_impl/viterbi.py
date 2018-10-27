import numpy as np
import time

def viterbi(observations, transitions, emissions, priors=None):
	num_states = emissions.shape[0]
	num_obs = len(observations)
	priors = priors if priors is not None else np.ones(num_states) / num_states

	ml_prob = np.zeros((num_states, num_obs))
	prev_state = np.zeros((num_states, num_obs), dtype=np.int)

	ml_prob[:, 0] = priors * emissions[:, 0]

	for i in range(1, num_obs):
		# ml_prob[:, i] = np.max(ml_prob[:, i-1] * transitions.T * emissions[np.newaxis, :, observations[i]].T, axis=1)
		# prev_state[:, i] = np.argmax(ml_prob[:, i-1] * transitions.T, axis=1)
		for j in range(num_states):
			ml_prob[j, i] = np.amax(ml_prob[:, i-1] * transitions[:, j] * emissions[j, observations[i]])
			prev_state[j, i] = np.argmax(ml_prob[:, i-1] * transitions[:, j] * emissions[j, observations[i]])

	states = np.zeros(num_obs, dtype=np.int)
	states[-1] = np.argmax(ml_prob[:, -1])
	for i in range(num_obs-1, 0, -1):
		# print(prev_state[states[i], i], states[i], i)
		states[i-1] = prev_state[states[i], i]

	print(ml_prob)
	print(prev_state)

	return states

if __name__ == '__main__':
	t = time.time()

	A = np.array([[.7, .3], [.4, .6]])
	B = np.array([[.5, .4, .1], [.1, .3, .6]])
	pi = [.6, .4]
	y = [0, 1, 2]
	answer = viterbi(y, A, B, pi)
	print(answer)

	A = np.array([[.75, .25], [.32, .68]])
	B = np.array([[.8, .1, .1], [.1, .2, .7]])
	y = np.array([0, 1, 2, 1, 0])
	answer = viterbi(y, A, B)
	print(answer) # internets said it should be [0, 1, 1, 1, 0]
	
	print("Ran in {}s".format(time.time()-t))