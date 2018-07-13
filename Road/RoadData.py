import math, random
import numpy as np
from PIL import Image, ImageDraw
import time, json
from scipy.stats import multivariate_normal
from Config import *
from scipy.ndimage.filters import gaussian_filter
import scipy

config = Config()

max_seq_len = config.MAX_NUM_VERTICES
blank = np.zeros(config.V_OUT_RES, dtype = np.uint8)
vertex_pool = [[] for _ in range(config.V_OUT_RES[1])]
for i in range(config.V_OUT_RES[1]):
	for j in range(config.V_OUT_RES[0]):
		vertex_pool[i].append(np.copy(blank))
		vertex_pool[i][j][i, j] = 255
		vertex_pool[i][j] = Image.fromarray(vertex_pool[i][j])
blank = Image.fromarray(blank)

roadJSON = json.load(open('../DataPreparation/RoadZurich.json'))
downsample = 8

class directed_graph(object):
	def __init__(self):
		self.v = []
		self.e = []
		self.nb = []
		return

	def add_v(self, v):
		self.v.append(v)
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

	def dfs(self, u):
		self.visited[u] = True
		for v, w in self.nb[u]:
			if not self.visited[v]:
				self.dfs(v)
		return

	def dijkstra(self, source):
		self.visited = [False for i in range(len(self.v))]
		self.dfs(source)
		Q = set([idx for idx, flag in enumerate(self.visited) if flag])
		dist = [np.inf for i in range(len(self.v))]
		prev = [None for i in range(len(self.v))]

		dist[source] = 0
		u = source
		while len(Q) > 0:
			Q.remove(u)
			for v, w in self.nb[u]:
				alt = dist[u] + w
				if alt < dist[v]:
					dist[v] = alt
					prev[v] = u
			d = np.inf
			for i in Q:
				if dist[i] < d:
					u = i
					d = dist[i]
		return dist, prev

	def dijkstra_all(self):
		self.sp = []
		for i in range(len(self.v)):
			self.sp.append(self.dijkstra(i))
		self.sp_max_idx = [np.argmax(dist) for dist, _ in self.sp]
		return

def make_ellipse(p, pad = 10):
	return [(p[0] - pad, p[1] - pad), (p[0] + pad, p[1] + pad)]

def getData(img_id, num_path, show = False):
	# img = scipy.misc.imread('../DataPreparation/RoadZurich/%s.png' % str(img_id).zfill(8))
	road = roadJSON[img_id]
	g = directed_graph()
	for item in road['v']:
		g.add_v(item)
	for s, t in road['e']:
		g.add_e(s, t)
	g.dijkstra_all()

	img = Image.open('../DataPreparation/RoadZurich/Zurich_%s.png' % str(img_id).zfill(8))
	draw = ImageDraw.Draw(img)
	for v in g.v:
		draw.ellipse(make_ellipse(v, pad = 6), fill = (255, 0, 0), outline = (255, 0, 0))
	for e in g.e:
		draw.line(g.v[e[0]] + g.v[e[1]], fill = (255, 0, 0), width = 5)
	if show:
		Image.fromarray(img).show()
		time.sleep(1)

	# Draw boundary and vertices
	boundary = Image.new('P', (w8, h8), color = 0)
	draw = ImageDraw.Draw(boundary)
	for e in g.e:
		draw.line(list(np.array(g.v[e[0]]) / downsample) + list(np.array(g.v[e[1]]) / downsample), fill = 255, width = 1)
	if show:
		boundary.show()
		time.sleep(1)
	boundary = np.array(boundary) / 255.0

	vertices = Image.new('P', (w8, h8), color = 0)
	draw = ImageDraw.Draw(vertices)
	for v in g.v:
		draw.ellipse(make_ellipse(list(np.array(v) / downsample), pad = 0), fill = 255, outline = 255)
	if show:
		vertices.show()
		time.sleep(1)
	vertices = np.array(vertices) / 255.0

	# RNN in and out
	vertex_terminals = []
	vertex_inputs = []
	vertex_outputs = []
	ends = []
	seq_lens = []
	for i in range(num_path):
		s = random.randint(0, len(g.v) - 1)
		t = g.sp_max_idx[s]
		dist, prev = g.sp[s]
		path = []
		p = t
		while p != s:
			path.append(p)
			p = prev[p]
		path.append(p)
		path.reverse()
		path_v = [np.array(g.v[idx]) / downsample for idx in path]
		vertex_input = [vertex_pool[int(r)][int(c)] for c, r in path_v]
		vertex_output = vertex_input[1:]
		vertex_terminal = [vertex_input[0], vertex_input[-1]]
		while len(vertex_input) < max_seq_len:
			vertex_input.append(blank)
		while len(vertex_output) < max_seq_len:
			vertex_output.append(blank)
		if show:
			for item in vertex_input:
				item.show()
				time.sleep(1)
		end = []
		for i in range(max_seq_len):
			if i < len(path_v) - 1:
				end.append(0)
			else:
				end.append(1)
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
	ids = np.random.choice(len(roadJSON), batch_size, replace = False)
	for i in range(batch_size):
		res.append(getData(ids[i], config.TRAIN_NUM_PATH, show))
	res = [np.array([item[i] for item in res]) for i in range(8)]
	if False:
		for item in res:
			print(item.shape)
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
	# print(len(peaks_with_score))
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
			ind = np.unravel_index(np.argmax(hmap), hmap.shape)
			if hmap[ind] >= end:
				path.append((ind[1] * 8, ind[0] * 8))
		pathImg = Image.new('P', (img.shape[1], img.shape[0]), color = 0)
		draw = ImageDraw.Draw(pathImg)
		draw.line(path, fill = 1, width = 5)
		res += np.array(pathImg, np.float32)
	res = np.array((res - res.min()) * 255.0 / (res.max() - res.min()), np.uint8)
	return res

if __name__ == '__main__':
	a = getDataBatch(5)
	b = getAllTerminal(a[2][0])
	print(b.shape)
	quit()
	c = recoverMultiPath(a[0][0], a[4][0])
	Image.fromarray(a[0][0]).show()
	Image.fromarray(c).show()

