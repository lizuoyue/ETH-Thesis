import math, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
from scipy.stats import multivariate_normal
from Config import *
from scipy.ndimage.filters import gaussian_filter

config = Config()

choices = [[0,1], [1,2], [2,3], [3,0], [0,1,2], [1,2,3], [2,3,0], [3,0,1], [0,1,2,3]]
choices = [[(i in item) for i in range(4)] for item in choices]
edge_choices = [(True, False), (False, True), (True, True)]

max_seq_len = config.MAX_NUM_VERTICES
blank = np.zeros(config.V_OUT_RES, dtype = np.uint8)
vertex_pool = [[] for _ in range(config.V_OUT_RES[1])]
for i in range(config.V_OUT_RES[1]):
	for j in range(config.V_OUT_RES[0]):
		vertex_pool[i].append(np.copy(blank))
		vertex_pool[i][j][i, j] = 255
		vertex_pool[i][j] = Image.fromarray(vertex_pool[i][j])
blank = Image.fromarray(blank)

class directed_graph(object):
	def __init__(self):
		self.v = []
		self.e = []
		self.d = {}
		self.nb = []
		return

	def add_v(self, v):
		self.d[v] = len(self.v)
		self.v.append(v)
		self.nb.append([])
		return

	def add_e(self, v1, v2, w = None, mode = 'val'):
		assert(mode == 'val' or mode == 'idx')
		if mode == 'val':
			if w is None:
				w = self.dist(v1, v2)
			self.e.append((self.d[v1], self.d[v2], w))
			self.nb[self.d[v1]].append((self.d[v2], w))
		if mode == 'idx':
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

	def dijkstra(self, source):
		Q = set(list(range(len(self.v))))
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

def pepper(img):
	row, col, ch = img.shape
	mean, var = 0, 1000
	gauss = np.random.normal(mean, var ** 0.5, img.shape)
	noisy = img + gauss
	noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
	return np.array(noisy * 255, np.uint8)

def getData(img_size, num_path, show = False):
	w , h  = img_size
	w2, h2 = int(w / 2.0), int(h / 2.0)
	w4, h4 = int(w / 4.0), int(h / 4.0)
	w8, h8 = int(w / 8.0), int(h / 8.0)
	downsample = 8

	base_pts = [
		(random.randint(w8     , w8 + w4), random.randint(h8     , h8 + h4)),
		(random.randint(w8 + w2, w  - w8), random.randint(h8     , h8 + h4)),
		(random.randint(w8 + w2, w  - w8), random.randint(h8 + h2, h  - h8)),
		(random.randint(w8     , w8 + w4), random.randint(h8 + h2, h  - h8)),
	]
	edge_pts = [
		[(random.randint(w8     , w8 + w4), 0    ), (0    , random.randint(h8     , h8 + h4))],
		[(random.randint(w8 + w2, w  - w8), 0    ), (w - 1, random.randint(h8     , h8 + h4))],
		[(random.randint(w8 + w2, w  - w8), h - 1), (w - 1, random.randint(h8 + h2, h  - h8))],
		[(random.randint(w8     , w8 + w4), h - 1), (0    , random.randint(h8 + h2, h  - h8))],
	]

	g = directed_graph()
	choice = choices[random.randint(0, len(choices) - 1)]
	for i in range(4):
		if choice[i]:
			g.add_v(base_pts[i])
			edge_choice = edge_choices[random.randint(0, len(edge_choices) - 1)]
			if edge_choice[0]:
				g.add_v(edge_pts[i][0])
				g.add_e(base_pts[i], edge_pts[i][0])
				g.add_e(edge_pts[i][0], base_pts[i])
			if edge_choice[1]:
				g.add_v(edge_pts[i][1])
				g.add_e(base_pts[i], edge_pts[i][1])
				g.add_e(edge_pts[i][1], base_pts[i])
	for i in range(4):
		if choice[i] and choice[i - 1]:
			g.add_e(base_pts[i], base_pts[i - 1])
			g.add_e(base_pts[i - 1], base_pts[i])
	g.dijkstra_all()
	# print(len(g.v))

	img = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(img)
	road_color = tuple(random.randint(64, 192) for _ in range(3))
	for v in g.v:
		draw.ellipse(make_ellipse(v, pad = 6), fill = road_color, outline = road_color)
	for e in g.e:
		draw.line(g.v[e[0]] + g.v[e[1]], fill = road_color, width = random.randint(10, 16))
	img = pepper(np.array(img))
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
	for i in range(batch_size):
		res.append(getData(config.AREA_SIZE, config.TRAIN_NUM_PATH, show))
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

def pil2np(image, show):
	if show:
		import matplotlib.pyplot as plt
	img = np.array(image, dtype = np.float32) / 255.0
	if len(img.shape) > 2 and img.shape[2] == 4:
		img = img[..., 0: 3]
	if show:
		plt.imshow(img)
		plt.show()
	return img

def plotPolygon(img_size = config.AREA_SIZE, num_vertices = 6, show = False):
	# Set image parameters
	num_row = img_size[0]
	num_col = img_size[1]
	half_x = math.floor(num_col / 2)
	half_y = math.floor(num_row / 2)
	img_size_s = (math.floor(num_row / 8), math.floor(num_col / 8))

	# Set polygon parameters
	epsilon = 1.0 / num_vertices
	center_r = math.floor(min(num_row, num_col) * 0.05) # <- Decide polygon's center
	polygon_size = math.floor(min(num_row, num_col) * 0.35) # <- Decide polygon's size
	delta_angle = np.pi * 2 * epsilon
	angle = np.random.uniform(0.0, delta_angle) # <- Decide polygon's first vertex

	# Determin the center of polygon
	center_x = half_x + np.random.randint(-center_r, center_r)
	center_y = half_y + np.random.randint(-center_r, center_r)

	# Determin the polygon vertices
	polygon = []
	polygon_s = []
	for i in range(num_vertices):
		r = polygon_size * np.random.uniform(0.8, 1.1) # <- Decide polygon's size range
		px = math.floor(center_x + r * np.cos(angle))
		py = math.floor(center_y - r * np.sin(angle)) # <- Decide polygon's order (counterclockwise)
		polygon.append((px, py))
		polygon_s.append((math.floor(px / 8), math.floor(py / 8)))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon) # <- Decide polygon's vertices
	first_idx = random.choice([i for i in range(num_vertices)])
	polygon = polygon[first_idx:] + polygon[:first_idx]
	polygon_s = polygon_s[first_idx:] + polygon_s[:first_idx]

	# Draw polygon
	color = config.TABLEAU20_DEEP[random.randint(0, 9)]
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	draw.polygon(polygon, fill = color, outline = color)

	# Add noise to the orginal image
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = Image.fromarray(img)
	img = pil2np(img, show)

	# Draw boundary
	boundary = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	boundary = pil2np(boundary, show)

	# Draw vertices
	vertices = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)
	vertices = pil2np(vertices, show)

	# Draw each vertex
	vertex_list = []
	for i in range(num_vertices):
		vertex = Image.new('P', img_size_s, color = 0)
		draw = ImageDraw.Draw(vertex)
		draw.point([polygon_s[i]], fill = 255)
		vertex = pil2np(vertex, show)
		vertex_list.append(vertex)
	# vertex_list.append(np.zeros(img_size_s, dtype = np.float32))
	# vertex_list = np.array(vertex_list)

	# Return
	if show:
		print(img.shape)
		print(boundary.shape)
		print(vertices.shape)
		print(np.array(vertex_list).shape)
	return img, boundary, vertices, vertex_list

def getDataBatchPolygon(batch_size):
	res = []
	num_v = np.random.choice(5, batch_size, replace = True) + 4
	for n in num_v:
		img, b, v, vertex_list = plotPolygon(num_vertices = n, show = False)
		while len(vertex_list) < config.MAX_NUM_VERTICES:
			vertex_list.append(np.zeros(config.V_OUT_RES, dtype = np.float32))
		vertex_outputs = vertex_list[1:] + [np.zeros(config.V_OUT_RES, dtype = np.float32)]
		vertex_terminals = [vertex_list[0]] * 2
		vertex_list = np.array(vertex_list)
		end = [0.0 for i in range(config.MAX_NUM_VERTICES)]
		end[n - 1] = 1.0
		end = np.array(end)
		res.append((img, b, v, [vertex_list], [vertex_outputs], [vertex_terminals], [end], [n]))
	return (np.array([item[i] for item in res]) for i in range(8))

if __name__ == '__main__':
	a = getDataBatchPolygon(4)
	for item in list(a):
		print(item.shape)
	quit()
	a = getDataBatch(1)
	b = getAllTerminal(a[2][0])
	print(b.shape)
	quit()
	c = recoverMultiPath(a[0][0], a[4][0])
	Image.fromarray(a[0][0]).show()
	Image.fromarray(c).show()

