import os, time, glob, sys
import numpy as np
import random, json
from scipy import spatial

cities = [
	'amsterdam', 'boston', 'denver', 'montreal', 'paris',
	'saltlakecity', 'toronto', 'kansas city', 'san diego', 'la'
] + ['pittsburgh', 'new york', 'tokyo', 'chicago', 'vancouver']

MAX_DIST = 1e12

class directed_graph(object):
	def __init__(self):
		self.v = []
		self.e = []
		self.nb = []
		self.dist = []
		return

	def add_v(self, v):
		self.v.append(v)
		self.nb.append([])
		self.dist.append([])
		return

	def add_e(self, v1, v2):
		assert(v1 in range(len(self.v)))
		assert(v2 in range(len(self.v)))
		w = self.l2_dist(self.v[v1], self.v[v2])
		self.e.append((v1, v2))
		self.nb[v1].append((v2, w))
		return

	def l2_dist(self, v1, v2):
		diff = np.array(v1) - np.array(v2)
		return np.sqrt(np.dot(diff, diff))

	def spfa(self, source):
		if not self.dist[source]:
			dist = [MAX_DIST for i in range(len(self.v))]
			# prev = [None for i in range(len(self.v))]
			in_q = [False for i in range(len(self.v))]
			dist[source] = 0
			q = [source]
			in_q[source] = True
			while len(q) > 0:
				u = q.pop(0)
				in_q[u] = False
				for v, w in self.nb[u]:
					alt = dist[u] + w
					if alt < dist[v]:
						dist[v] = alt
						# prev[v] = u
						if not in_q[v]:
							in_q[v] = True
							q.append(v)
			self.dist[source] = dist
		return self.dist[source]#, prev

	def make_kd_tree(self):
		self.kd_tree = spatial.KDTree(self.v)

def read_graph(filename):
	with open(filename) as f:
		lines = f.readlines()
	for i, line in enumerate(lines):
		if line == '\n':
			sep = i
			break
	v = lines[:sep]
	e = lines[sep+1:]
	v = [tuple(float(item) for item in line.strip().split()) for line in v]
	e = [tuple(int(item) for item in line.strip().split()) for line in e]

	dg = directed_graph()
	for vv in v:
		dg.add_v(vv)
	for v1, v2 in e:
		dg.add_e(v1, v2)
	return dg

if __name__ == '__main__':
	gt_by_city = {}
	for city in cities:
		gt_by_city[city] = read_graph('dataset/data/graphs/%s.graph' % city)
		gt_by_city[city].make_kd_tree()

	# start_by_city = eval(open('starting_location.txt').read()))
	start_by_city = {}
	for seed in [6666, 6678, 6679, 6687, 6708, 8888, 8866, 8686, 8666, 8688]:
		np.random.seed(seed)
		for city in cities:
			li = list(np.random.choice(len(gt_by_city[city].v), 10, replace = False))
			if city in start_by_city:
				start_by_city[city].extend(li)
			else:
				start_by_city[city] = li
	for file in [
		'out_roadtracer/%s.0.3_0.3.newreconnect.graph',
		'out_ours_cu/%s_0.70.out.graph',
		'out_ours_uc/%s_0.70.out.graph',
		'out_deep/%s.fix.connected.graph',
		# 'segmentation/thr10/%s.pix.graph',
		# 'segmentation/thr20/%s.pix.graph',
		# 'segmentation/thr30/%s.pix.graph',
		# 'segmentation/thr40/%s.pix.graph',
	]:
		print(file)
		rec_li = []
		for city in cities:
			gt = gt_by_city[city]
			dt = read_graph(file % city)
			dt.make_kd_tree()
			start_pairs = []
			gt_start_set, dt_start_set = set(), set()
			for item in start_by_city[city]:
				dist, idx = dt.kd_tree.query(gt.v[item])
				start_pairs.append((item, idx, dist))
				gt_start_set.add(item)
				dt_start_set.add(idx)

			end_li = gt.kd_tree.query_ball_tree(dt.kd_tree, r = 20)
			end_pairs = []
			for gt_idx, indices in enumerate(end_li):
				for dt_idx in indices:
					if gt_idx not in gt_start_set and dt_idx not in dt_start_set:
						end_pairs.append((gt_idx, dt_idx))
			end_choice = np.random.choice(len(end_pairs), 1000, replace = len(end_pairs) < 1000)
			end_pairs = [end_pairs[i] for i in end_choice]

			for gts, dts, d_start in start_pairs:
				for gte, dte in end_pairs:
					d_end = gt.l2_dist(gt.v[gte], dt.v[dte])
					gtd = gt.spfa(gts)[gte] + 0#(d_start + d_end) / 2
					dtd = dt.spfa(dts)[dte] + (d_start + d_end) / 2 * 2
					if gtd > 1e9 or dtd > 1e9:
						continue
					rec_li.append([gtd, dtd])
		rec_li = np.array(rec_li)
		gtd, dtd = rec_li[:, 0], rec_li[:, 1]
		w_gt, w_dt = gtd / gtd.max(), dtd / dtd.max()
		recs = []
		for th in [0.05, 0.10, 0.15]:
			score = np.logical_and(gtd * (1 - th) <= dtd, dtd <= gtd * (1 + th))
			recs.append(np.dot(w_gt, score) / np.sum(w_gt))
		print(file, 'rec range', recs)
		recs = []
		for th in [0.95, 0.90, 0.85]:
			score = (np.minimum(gtd, dtd) / np.maximum(gtd, dtd)) >= th
			recs.append(np.dot(w_gt, score) / np.sum(w_gt))
		print(file, 'rec iou', recs)


		pre_li = []
		for city in cities:
			gt = gt_by_city[city]
			dt = read_graph(file % city)
			dt.make_kd_tree()
			start_pairs = []
			gt_start_set, dt_start_set = set(), set()
			for item in start_by_city[city]:
				dist, idx = dt.kd_tree.query(gt.v[item])
				start_pairs.append((item, idx, dist))
				gt_start_set.add(item)
				dt_start_set.add(idx)

			end_li = dt.kd_tree.query_ball_tree(gt.kd_tree, r = 20)
			end_pairs = []
			for dt_idx, indices in enumerate(end_li):
				for gt_idx in indices:
					if gt_idx not in gt_start_set and dt_idx not in dt_start_set:
						end_pairs.append((gt_idx, dt_idx))
			end_choice = np.random.choice(len(end_pairs), 1000, replace = len(end_pairs) < 1000)
			end_pairs = [end_pairs[i] for i in end_choice]

			for gts, dts, d_start in start_pairs:
				for gte, dte in end_pairs:
					d_end = gt.l2_dist(gt.v[gte], dt.v[dte])
					gtd = gt.spfa(gts)[gte] + (d_start + d_end) / 2 * 2
					dtd = dt.spfa(dts)[dte] + 0 # (d_start + d_end) / 2
					if gtd > 1e9 or dtd > 1e9:
						continue
					pre_li.append([gtd, dtd])
		pre_li = np.array(pre_li)
		gtd, dtd = pre_li[:, 0], pre_li[:, 1]
		w_gt, w_dt = gtd / gtd.max(), dtd / dtd.max()
		pres = []
		for th in [0.05, 0.10, 0.15]:
			score = np.logical_and(gtd * (1 - th) <= dtd, dtd <= gtd * (1 + th))
			pres.append(np.dot(w_dt, score) / np.sum(w_dt))
		print(file, 'pre range', pres)
		pres = []
		for th in [0.95, 0.90, 0.85]:
			score = (np.minimum(gtd, dtd) / np.maximum(gtd, dtd)) >= th
			pres.append(np.dot(w_dt, score) / np.sum(w_dt))
		print(file, 'pre iou', pres)





