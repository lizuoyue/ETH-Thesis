import math, random
import numpy as np
from PIL import Image, ImageDraw
import time, json
from scipy.stats import multivariate_normal
from Config import *
from scipy.ndimage.filters import gaussian_filter
import scipy, socket, sys

config = Config()

if socket.gethostname() == 'cnb-d102-50':
	file_path = '../DataPreparation'
else:
	file_path = '/cluster/scratch/zoli/road'
max_seq_len = config.MAX_NUM_VERTICES
blank = np.zeros(config.V_OUT_RES, dtype = np.uint8)
vertex_pool = [[] for _ in range(config.V_OUT_RES[1])]
for i in range(config.V_OUT_RES[1]):
	for j in range(config.V_OUT_RES[0]):
		vertex_pool[i].append(np.copy(blank))
		vertex_pool[i][j][i, j] = 255
		vertex_pool[i][j] = Image.fromarray(vertex_pool[i][j])
blank = Image.fromarray(blank)

city_name = sys.argv[1]

roadJSON = json.load(open(file_path + '/Road%s.json' % city_name))
downsample = 8

np.random.seed(6666)
mini_ids = np.random.choice(len(roadJSON), 20, replace = False)

class disjoint_set(object):
	def __init__(self, num = 0):
		self.parent = list(range(num))
		self.rank = [0] * num
		return

	def make_set(self, num):
		for _ in range(num):
			self.parent.append(len(self.parent))
			self.rank.append(0)
		return

	def find(self, x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]

	def union(self, x, y):
		xRoot = self.find(x)
		yRoot = self.find(y)
		if xRoot == yRoot:
			return
		if self.rank[xRoot] < self.rank[yRoot]:
			self.parent[xRoot] = yRoot
		else:
			self.parent[yRoot] = xRoot
			if self.rank[xRoot] == self.rank[yRoot]:
				self.rank[xRoot] += 1
		return

	def get_set_by_id(self):
		for i in range(len(self.parent)):
			self.find(i)
		d = {}
		for i in range(len(self.parent)):
			if self.parent[i] in d:
				d[self.parent[i]].append(i)
			else:
				d[self.parent[i]] = [i]
		return [d[k] for k in d]

class directed_graph(object):
	def __init__(self, downsample = 8):
		self.v = []
		self.v_org = []
		self.e = []
		self.nb = []
		return

	def add_v(self, v):
		self.v.append(v)
		self.v_org.append((v[0] * 8 + 4, v[1] * 8 + 4))
		self.nb.append([])
		return

	def add_e(self, v1, v2, w = None):
		assert(v1 in range(len(self.v)))
		assert(v2 in range(len(self.v)))
		if w is None:
			w = self.dist(self.v[v1], self.v[v2])
		self.e.append((v1, v2, w))
		self.nb[v1].append((v2, w))
		return

	def dist(self, v1, v2):
		diff = np.array(v1) - np.array(v2)
		return np.sqrt(np.dot(diff, diff))

	def spfa(self, source):
		dist = [1e9 for i in range(len(self.v))]
		prev = [None for i in range(len(self.v))]
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
					prev[v] = u
					if not in_q[v]:
						in_q[v] = True
						q.append(v)
		dist = np.array(dist)
		dist[dist > 1e8] = -1e9
		return dist, prev

	def shortest_path_all(self):
		self.sp = []
		for i in range(len(self.v)):
			self.sp.append(self.spfa(i))
		self.sp_max_idx = [np.argmax(dist) for dist, _ in self.sp]
		return

def make_ellipse(p, pad = 10):
	return [(p[0] - pad, p[1] - pad), (p[0] + pad, p[1] + pad)]

def colinear(p0, p1, p2):
	x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
	x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
	return abs(x1 * y2 - x2 * y1) < 1e-6

def path_processing(g, path):
	path_v = [g.v[idx] for idx in path]
	# return path_v
	# deg = [len(g.nb[idx]) for idx in path]
	# rep = [path_v[k] == path_v[k - 1] for k in range(len(path))]
	# lin = [colinear(path_v[k - 1], path_v[k], path_v[(k + 1) % len(path)]) for k in range(len(path))]
	# new_path_v = [item for k, item in enumerate(path_v) if (not rep[k]) and (not (deg[k] == 2 and lin[k]))]
	# if sum([y or (x == 2 and z) for x, y, z in zip(deg, rep, lin)]) > 0:
	# 	print(path_v)
	# 	print(new_path_v)
	# 	input()
	new_path_v = path_v[: max_seq_len]
	return new_path_v

def getData(img_id, num_path, show = False):
	################## Preprocessing ##################
	# 1. Remove duplicate
	road = roadJSON[img_id]
	v_val = [tuple(np.floor(np.array(v) / 599.0 * (config.AREA_SIZE[0] - 1) / downsample).astype(np.int32)) for v in road['v']]
	e_val = [(v_val[s], v_val[t]) for s, t in road['e']]
	v_val = list(set(v_val))
	e_val = list(set(e_val))
	v_val2idx = {v: k for k, v in enumerate(v_val)}
	e_idx = [(v_val2idx[s], v_val2idx[t]) for s, t in e_val]
	e_idx = [(s, t) for s, t in e_idx if s != t]

	# 2. Get v to be removed
	nb = [[] for _ in range(len(v_val))]
	for s, t in e_idx:
		nb[s].append(t)
	v_rm = []
	for vid, (v, vnb) in enumerate(zip(v_val, nb)):
		if len(vnb) == 2:
			v0, v1 = v_val[vnb[0]], v_val[vnb[1]]
			if colinear(v, v0, v1):
				v_rm.append(v_val2idx[v])
	v_rm_set = set(v_rm)
	if len(v_rm_set) < 0:
		show = True

	# 3. Get e to be added
	e_add = []
	visited = [False for _ in range(len(v_val))]
	for vid in v_rm_set:
		if not visited[vid]:
			visited[vid] = True
			assert(len(nb[vid]) == 2)
			res = []
			for nvid_iter in nb[vid]:
				nvid = int(nvid_iter)
				while nvid in v_rm_set:
					visited[nvid] = True
					v1, v2 = nb[nvid]
					assert((v1 in v_rm_set and visited[v1]) + (v2 in v_rm_set and visited[v2]) == 1)
					if (v1 in v_rm_set and visited[v1]):
						nvid = v2
					else:
						nvid = v1
				res.append(nvid)
			assert(len(res) == 2)
			e_add.append((res[0], res[1]))
			e_add.append((res[1], res[0]))

	# 4. Remove v and add e
	e_idx = [(s, t) for s, t in e_idx if s not in v_rm_set and t not in v_rm_set]
	e_idx.extend(e_add)
	e_val = [(v_val[s], v_val[t]) for s, t in e_idx]
	v_val = [v for i, v in enumerate(v_val) if i not in v_rm_set]
	v_val2idx = {v: k for k, v in enumerate(v_val)}
	e_idx = [(v_val2idx[s], v_val2idx[t]) for s, t in e_val]
	###################################################

	g = directed_graph()
	for v in v_val:
		g.add_v(v)
	for s, t in e_idx:
		g.add_e(s, t)
	g.shortest_path_all()

	img = Image.open(file_path + '/Road%s/%s_%s.png' % (city_name, city_name, str(img_id).zfill(8))).resize(config.AREA_SIZE)
	w8, h8 = img.size
	w8 = int(w8 / float(downsample))
	h8 = int(h8 / float(downsample))

	if show:
		draw = ImageDraw.Draw(img)
		for v in g.v_org:
			draw.ellipse(make_ellipse(v, pad = 5), fill = (255, 0, 0), outline = (255, 0, 0))
		for e in g.e:
			draw.line(g.v_org[e[0]] + g.v_org[e[1]], fill = (255, 0, 0), width = 2)
		img.show()
		time.sleep(0.1)
	img = np.array(img)[..., 0: 3]

	# Draw boundary and vertices
	boundary = Image.new('P', (w8, h8), color = 0)
	draw = ImageDraw.Draw(boundary)
	for e in g.e:
		draw.line(list(g.v[e[0]]) + list(g.v[e[1]]), fill = 255, width = 1)
	if show:
		boundary.resize(config.AREA_SIZE).show()
		time.sleep(0.1)
	boundary = np.array(boundary) / 255.0

	vertices = Image.new('P', (w8, h8), color = 0)
	draw = ImageDraw.Draw(vertices)
	for i in range(len(g.v)):
		draw.ellipse(make_ellipse(g.v[i], pad = 0), fill = 255, outline = 255)
	if show:
		vertices.resize(config.AREA_SIZE).show()
		time.sleep(0.1)
	vertices = np.array(vertices) / 255.0

	###########
	ddd, s_chosen = -1e9, 0
	for s in range(len(g.v)):
		tmp_d = g.sp[s][0][g.sp_max_idx[s]]
		if tmp_d > ddd:
			s_chosen = s
			ddd = tmp_d
	###########

	# RNN in and out
	vertex_terminals = []
	vertex_inputs = []
	vertex_outputs = []
	ends = []
	seq_lens = []
	for i in range(num_path):
		path = []
		if len(g.v) > 0:
			# if i < len(g.v):
			# 	s = i
			# else:
			# 	s = random.randint(0, len(g.v) - 1)
			s = s_chosen
			t = g.sp_max_idx[s]
			dist, prev = g.sp[s]
			p = t
			while p != s:
				path.append(p)
				p = prev[p]
			path.append(p)
			path.reverse()
		path_v = path_processing(g, path)
		vertex_input = [vertex_pool[r][c] for c, r in path_v]
		if len(vertex_input) > 0:
			vertex_output = vertex_input[1:]
			vertex_terminal = [vertex_input[0], vertex_input[-1]]
		else:
			vertex_output = []
			vertex_terminal = [blank, blank]
		while len(vertex_input) < max_seq_len:
			vertex_input.append(blank)
		while len(vertex_output) < max_seq_len:
			vertex_output.append(blank)
		if len(vertex_input) != max_seq_len:
			print(len(vertex_input))
		assert(len(vertex_output) == max_seq_len)
		end = [0 for i in range(max_seq_len)]
		if len(path_v) > 0:
			end[len(path_v) - 1] = 1

		if False:
			color = [0] + [1, 2] * 30
			for vvv in [vertex_input, vertex_output, vertex_terminal]:
				visualize = np.zeros((config.V_OUT_RES[1], config.V_OUT_RES[0], 3), np.uint8)
				for i, item in enumerate(vvv):
					visualize[..., color[i]] = np.maximum(visualize[..., color[i]], np.array(item, np.uint8))
				Image.fromarray(visualize).resize(config.AREA_SIZE).show()
				time.sleep(0.1)
			print(end)
			print(len(path_v))
			# input()

		vertex_input = [np.array(item) / 255.0 for item in vertex_input]
		vertex_output = [np.array(item) / 255.0 for item in vertex_output]
		vertex_terminal = [np.array(item) / 255.0 for item in vertex_terminal]
		vertex_inputs.append(vertex_input)
		vertex_outputs.append(vertex_output)
		vertex_terminals.append(vertex_terminal)
		ends.append(end)
		seq_lens.append(len(path_v))
	vertex_inputs = np.array(vertex_inputs)
	vertex_outputs = np.array(vertex_outputs)
	vertex_terminals = np.array(vertex_terminals)
	ends = np.array(ends)
	seq_lens = np.array(seq_lens)

	# print(img.shape)
	# print(boundary.shape)
	# print(vertices.shape)
	# print(vertex_inputs.shape)
	# print(vertex_outputs.shape)
	# print(vertex_terminals.shape)
	# print(ends.shape)
	# print(seq_lens.shape)

	return img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens

def getDataBatch(batch_size, show = False):
	res = []
	ids = np.random.choice(len(mini_ids), batch_size, replace = False)
	for i in range(batch_size):
		res.append(getData(mini_ids[ids[i]], config.TRAIN_NUM_PATH, show))
	res = [np.array([item[i] for item in res]) for i in range(8)]
	if False:
		for item in res:
			np.array(item).shape
	return res

def findPeaks(heatmap, sigma = 0):
	th = 0
	hmap = gaussian_filter(heatmap, sigma)
	map_left = np.zeros(hmap.shape)
	map_left[1:,:] = hmap[:-1,:]
	map_right = np.zeros(hmap.shape)
	map_right[:-1,:] = hmap[1:,:]
	map_up = np.zeros(hmap.shape)
	map_up[:,1:] = hmap[:,:-1]
	map_down = np.zeros(hmap.shape)
	map_down[:,:-1] = hmap[:,1:]
	map_ul = np.zeros(hmap.shape)
	map_ul[1:,1:] = hmap[:-1,:-1]
	map_ur = np.zeros(hmap.shape)
	map_ur[:-1,1:] = hmap[1:,:-1]
	map_dl = np.zeros(hmap.shape)
	map_dl[1:,:-1] = hmap[:-1,1:]
	map_dr = np.zeros(hmap.shape)
	map_dr[:-1,:-1] = hmap[1:,1:]
	summary = np.zeros(hmap.shape)
	summary += hmap>=map_left+th
	summary += hmap>=map_right+th
	summary += hmap>=map_up+th
	summary += hmap>=map_down+th
	summary += hmap>=map_dl+th
	summary += hmap>=map_dr+th
	summary += hmap>=map_ul+th
	summary += hmap>=map_ur+th
	peaks_binary = np.logical_and.reduce((summary >= 8, hmap >= 0.7))
	peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
	peaks_with_score = [x + (heatmap[x[1],x[0]],) for x in peaks]
	return peaks_with_score

def getAllTerminal(hmap):
	res = []
	peaks_with_score = findPeaks(hmap)
	print(peaks_with_score)
	for i in range(len(peaks_with_score)):
		x1, y1, _ = peaks_with_score[i]
		for j in range(len(peaks_with_score)):
			if j == i:
				continue
			x2, y2, _ = peaks_with_score[j]
			res.append([np.array(vertex_pool[y1][x1]), np.array(vertex_pool[y2][x2])])
	return np.array(res)

def recoverMultiPath(img, paths):
	res = np.zeros((img.shape[0], img.shape[1]))
	for i in range(paths.shape[0]):
		path = []
		for j in range(max_seq_len):
			hmap = paths[i, j]
			end = 1 - hmap.sum()
			print(hmap.sum(), end)
			ind = np.unravel_index(np.argmax(hmap), hmap.shape)
			if hmap[ind] >= end:
				path.append((ind[1] * 8 + 4, ind[0] * 8 + 4))
			else:
				break
		pathImg = Image.new('P', (img.shape[1], img.shape[0]), color = 0)
		draw = ImageDraw.Draw(pathImg)
		draw.line(path, fill = 1, width = 5)
		res += np.array(pathImg, np.float32)
	res = np.array((res - res.min()) * 255.0 / (res.max() - res.min() + 1e-9), np.uint8)
	return res

if __name__ == '__main__':
	for _ in range(1000):
		img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens = getDataBatch(10, show = False)
		print(vertex_outputs.sum(axis = -1).sum(axis = -1) + ends)
		print(seq_lens)
		quit()
	# b = getAllTerminal(a[2][0])
	# print(b.shape)
	# quit()
	# c = recoverMultiPath(a[0][0], a[4][0])
	# Image.fromarray(a[0][0]).show()
	# Image.fromarray(c).show()

