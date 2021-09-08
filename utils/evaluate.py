from __future__ import absolute_import, division, print_function
import numpy as np


def calc_dists(preds, target, normalize):
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0]))

	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds   =  preds[n, c, :] #/ normalize[n]
				normed_targets = target[n, c, :] #/ normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
				#dists[c, n]    = np.sqrt((preds[n, c, 0] - target[n, c, 0]) ** 2 + (preds[n, c, 1] - target[n, c, 1]) ** 2)
			else:
				dists[c, n]    = -1

	return dists


def dist_acc(dists, thresh):
	#print(thresh)
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()
	#print(np.less(dists[dist_cal], thresh))
	#print(thresh)

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], thresh).sum() * 1.0 / num_dist_cal
	else:
		return -1


def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(heatmaps_reshaped, 2)
	maxvals           = np.amax(heatmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask

	return preds, maxvals



def accuracy(output, target, thr_PCK, thr_PCKh, dataset, width, hm_type='gaussian', threshold=0.5):
	idx  = list(range(output.shape[1]))
	#print(output.shape)
	norm = 1.0

	if hm_type == 'gaussian':
		pred, _   = get_max_preds(output)
		target, _ = get_max_preds(target)

		h         = output.shape[2]
		w         = output.shape[3]
		norm      = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10
		#print(pred.shape[1])
		#print(pred.shape[0])

	dists = calc_dists(pred, target, norm)
	#print(dists)

	acc     = np.zeros((len(idx)))
	avg_acc = 0
	cnt     = 0
	visible = np.zeros((len(idx)))

	for i in range(len(idx)):
		acc[i] = 1 #dist_acc(dists[idx[i]], thr_PCK*(width/2.2 ))
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	bbox = w / 2.2
	avg_PCKh = 0


	for i in range(len(idx)):
		PCKh[i] = 1 #dist_acc(dists[idx[i]], thr_PCKh*bbox)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh


	# PCK
	PCK = np.zeros((len(idx)))
	#print(width)
	bbox = width / 2.2
	avg_PCK = 0
	#print(dists)
	#print(width)
	#print(thr_PCK*bbox)
	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], thr_PCK*bbox)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0
	#print(PCK.shape)

	#if cnt != 0:
#		PCK[0] = avg_PCK


	return acc, PCK, PCKh, cnt, pred, visible