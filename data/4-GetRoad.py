import math, sys, time, os, io, requests, json, glob, random
import numpy as np
from PIL import Image, ImageDraw
from UtilityGeography import BoundingBox
from Config import Config

config = Config()

class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
			np.int16, np.int32, np.int64, np.uint8,
			np.uint16, np.uint32, np.uint64)):
			return int(obj)
		elif isinstance(obj, (np.float_, np.float16, np.float32, 
			np.float64)):
			return float(obj)
		elif isinstance(obj,(np.ndarray,)):
			print(obj.tolist())
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def area(polygon):
	s = 0.0
	for i in range(len(polygon)):
		x1, y1 = polygon[i - 1]
		x2, y2 = polygon[i]
		s += (x1 * y2 - x2 * y1)
	s /= 2.0
	return s

def colinear(p0, p1, p2):
	x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
	x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
	return abs(x1 * y2 - x2 * y1) < 1e-6

def colinear_angle(p0, p1, p2):
	def l2dist(a, b):
		diff = np.array(a) - np.array(b)
		return np.sqrt(np.dot(diff, diff))
	li = [l2dist(p0, p1), l2dist(p1, p2), l2dist(p0, p2)]
	li.sort()
	a, b, c = li
	cos_C = (a * a + b * b - c * c) / (2 * a * b)
	return cos_C < -0.996 # cos(174.8736°)

def clip(subjectPolygon, clipPolygon):
	# both polygons should be clockwise/anti-clockwise
	def inside(p):
		return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
	def computeIntersection():
		dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
		dp = [ s[0] - e[0], s[1] - e[1] ]
		n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
		n2 = s[0] * e[1] - s[1] * e[0] 
		n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
		return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

	if area(subjectPolygon) < 0:
		subjectPolygon.reverse()
	if area(clipPolygon) < 0:
		clipPolygon.reverse()
 
	outputList = subjectPolygon
	cp1 = clipPolygon[-1]
 
	for clipVertex in clipPolygon:
		cp2 = clipVertex
		inputList = outputList
		outputList = []
		if len(inputList) == 0:
			return []
		s = inputList[-1] 
		for subjectVertex in inputList:
			e = subjectVertex
			if inside(e):
				if not inside(s):
					outputList.append(computeIntersection())
				outputList.append(e)
			elif inside(s):
				outputList.append(computeIntersection())
			s = e
		cp1 = cp2
	return outputList

def clip_in_img(subjectPolygon, w, h):
	clipPolygon = [(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)]
	subjectPolygon = [[x, -y] for x, y in subjectPolygon]
	clipPolygon = [[x, -y] for x, y in clipPolygon]
	res = clip(subjectPolygon, clipPolygon)
	return [(round(x), round(-y)) for x, y in res]

def saveEdgeImg(edges, size, filename):
	img = Image.new('P', size, color = 255)
	draw = ImageDraw.Draw(img)
	for v1, v2 in edges:
		for xx, yy in [v1, v2]:
			draw.rectangle([xx - 5, yy - 5, xx + 5, yy + 5], fill = 0)
		draw.line(list(v1) + list(v2), fill = 0, width = 5)
	img.save(filename)
	return

def savePolygonImg(polygon, size, filename):
	img = Image.new('P', size, color = 255)
	draw = ImageDraw.Draw(img)
	draw.polygon(polygon, fill = 255, outline = 0)
	img.save(filename)

class RoadPool(object):
	def __init__(self):
		self.v = {}
		self.e = []
		self.bbox = []
		self.opt = ['l', 'u', 'r', 'd']

	def addV(self, vid, info):
		if vid in self.v:
			assert(info == self.v[vid])
		else:
			self.v[vid] = info
		return

	def addE(self, vid1, vid2):
		lon1, lat1 = self.v[vid1]
		lon2, lat2 = self.v[vid2]
		self.e.append((vid1, vid2))
		self.bbox.append([
			min(lon1, lon2),
			max(lat1, lat2),
			max(lon1, lon2),
			min(lat1, lat2)
		])
		return

	def sortV(self):
		assert(len(self.e) == len(self.bbox))
		self.bidSet = set([i for i in range(len(self.e))])
		self.bSorted = {}
		self.bSortedID = {}
		for i, opt in enumerate(self.opt):
			self.bSorted[opt] = [(item[i], bid) for bid, item in enumerate(self.bbox)]
			self.bSorted[opt].sort()
			self.bSortedID[opt] = [item[1] for item in self.bSorted[opt]]
		self.minVal = {opt: self.bSorted[opt][ 0][0] for opt in self.opt}
		self.maxVal = {opt: self.bSorted[opt][-1][0] for opt in self.opt}
		return

	def _findB_G(self, opt, th):
		# return index in self.bSorted
		# find the first one (with min idx) whose val > th
		li = self.bSorted[opt]
		if th < self.minVal[opt]:
			return 0
		if th >= self.maxVal[opt]:
			return len(li)
		l, r = 0, len(li) - 1
		while l < r:
			mid = int(math.floor((l + r) / 2))
			if li[mid][0] >= th:
				r = mid
			else:
				l = mid + 1
		return l

	def _findB_L(self, opt, th):
		# return index in self.bSorted
		# find the last one (with max idx) whose val < th
		li = self.bSorted[opt]
		if th <= self.minVal[opt]:
			return -1
		if th > self.maxVal[opt]:
			return len(li) - 1
		l, r = 0, len(li) - 1
		while l < r:
			mid = int(math.ceil((l + r) / 2))
			if li[mid][0] <= th:
				l = mid
			else:
				r = mid - 1
		return l

	def findV(self, minLon, maxLon, minLat, maxLat):
		assert(minLon <= maxLon)
		assert(minLat <= maxLat)

		maxLonIdx = self._findB_G('l', maxLon)
		minLatIdx = self._findB_L('u', minLat)
		minLonIdx = self._findB_L('r', minLon)
		maxLatIdx = self._findB_G('d', maxLat)

		res = self.bidSet \
			.difference(self.bSortedID['l'][maxLonIdx:]) \
			.difference(self.bSortedID['u'][:minLatIdx + 1]) \
			.difference(self.bSortedID['r'][:minLonIdx + 1]) \
			.difference(self.bSortedID['d'][maxLatIdx:]) \

		return res

def graphProcess(graph):
	# graph: [(edge_1), ..., (edge_n)]
	## edge: ((x1, y1), (x2, y2))

	# 1. Remove duplicate
	v_val, e_val = set(), set()
	for item in graph:
		v_val.add(item[0])
		v_val.add(item[1])
		e_val.add(item)
		e_val.add((item[1], item[0]))
	v_val = list(v_val)
	e_val = list(e_val)
	v_val2idx = {v: k for k, v in enumerate(v_val)}
	e_idx = [(v_val2idx[s], v_val2idx[t]) for s, t in e_val]

	# 2. Get v to be removed
	nb = [[] for _ in range(len(v_val))]
	for s, t in e_idx:
		nb[s].append(t)
	v_rm = []
	for vid, (v, vnb) in enumerate(zip(v_val, nb)):
		if len(vnb) == 2:
			v0, v1 = v_val[vnb[0]], v_val[vnb[1]]
			if colinear_angle(v, v0, v1):
				v_rm.append(vid)
	v_rm_set = set(v_rm)

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
					if v1 in v_rm_set and visited[v1]:
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

	return e_val

def extractPolygons(edges):
	eSet = set()
	nb = {}
	for u, v in edges:
		eSet.add((u, v))
		eSet.add((v, u))
		nb[u] = set()
		nb[v] = set()
	for u, v in eSet:
		nb[u].add(v)
		nb[v].add(u)
	res = []
	print(eSet)
	while len(eSet) > 0:
		random.seed(8888)
		v_start, v_next = random.sample(eSet, 1)[0]
		print(v_start)
		v_prev, v_now = None, v_start
		polygon = [v_now]
		while v_next != v_start:
			polygon.append(v_next)
			eSet.remove((v_now, v_next))
			nb[v_now].remove(v_next)
			v_prev = v_now
			v_now = v_next
			vec1 = np.array(v_now) - np.array(v_prev)
			comp = []
			for v in nb[v_now]:
				vec2 = np.array(v) - np.array(v_now)
				cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
				comp.append((cross, v))
			_, v_next = max(comp)
		res.append(polygon)
	return res


def cropMap(road_pool, map_info, mid, city_info, patch_seq, ann_seq):
	city_name = city_info['city_name']
	dx, dy = city_info['r_step']
	bw, bh = city_info['r_size']
	x1, x2 = city_info['r_x_range']
	y1, y2 = city_info['r_y_range']

	info = map_info[mid]
	c_lat, c_lon = info['center']
	w, h = info['size']
	z, s = info['zoom'], info['scale']
	idx = city_info['val_test'](c_lon, c_lat)

	patches, roads = [], []
	map_img = np.array(Image.open('./%sMap/%s.png' % (city_name, str(mid).zfill(6))))
	map_box = BoundingBox(c_lon, c_lat, (w - config.PAD * 2) * s, (h - config.PAD * 2) * s, z, s)
	for x in range(x1, x2):
		for y in range(y1, y2):
			l, u = map_box.c_rpx + x * dx - int(bw / 2), map_box.c_rpy + y * dy - int(bh / 2)
			r, d = l + bw, u + bh
			patch_file = './%sPatch/%s.png' % (city_name, str(patch_seq).zfill(6))
			if not os.path.exists(patch_file):
				Image.fromarray(map_img[u: d, l: r, ...]).save(patch_file)
			minLon, maxLat = map_box.relativePixelToLonLat(l, u)
			maxLon, minLat = map_box.relativePixelToLonLat(r, d)
			tmp_clon, tmp_clat = (minLon + maxLon) / 2, (minLat + maxLat) / 2
			tmp_box = BoundingBox(tmp_clon, tmp_clat, bw, bh, z, s)

			patch = {}
			patch['id'] = patch_seq
			patch['file_name'] = '%s.png' % str(patch_seq).zfill(6)
			patch['width'] = bw
			patch['height'] = bh

			patch['map_id'] = mid
			patch['map_crop_lurd'] = [l, u, r, d]
			patch['box_lon_lat'] = [minLon, maxLat, maxLon, minLat]
			patches.append(patch)

			road = {'category_id': 999999, 'iscrowd': 0}
			road['id'] = ann_seq
			road['image_id'] = patch_seq
			road['area'] = 0
			road['bbox'] = [0, 0, 0, 0]

			eSet = set()
			bids = road_pool.findV(minLon, maxLon, minLat, maxLat)
			for bid in bids:
				vid1, vid2 = road_pool.e[bid]
				lon1, lat1 = road_pool.v[vid1]
				lon2, lat2 = road_pool.v[vid2]
				xx1, yy1 = tmp_box.lonLatToRelativePixel(lon1, lat1)
				xx2, yy2 = tmp_box.lonLatToRelativePixel(lon2, lat2)
				flag1 = 0 <= xx1 and xx1 < bw
				flag2 = 0 <= yy1 and yy1 < bh
				flag3 = 0 <= xx2 and xx2 < bw
				flag4 = 0 <= yy2 and yy2 < bh
				seg = [(xx1, yy1), (xx2, yy2)]
				if flag1 and flag2 and flag3 and flag4:
					res = seg
				else:
					res = clip_in_img(seg, bw, bh)
				flag = [res[i] == res[i - 1] for i in range(len(res))]
				res = [item for item, f in zip(res, flag) if not f]
				if len(res) == 2:
					res.sort()
					eSet.add(tuple(res))
				elif len(res) == 1:
					res.append((xx2, yy2))
					res.sort()
					eSet.add(tuple(res))
				else:
					assert(len(res) == 0)

			for u, v in list(eSet):
				eSet.add((v, u))

			road['segmentation'] = graphProcess(list(eSet))

			if False: # Show diff before/after removal of colinear
				if eSet != set(road['segmentation']):
					saveEdgeImg(eSet, (bw, bh), '%sRoad1.png' % str(patch_seq).zfill(6))
					saveEdgeImg(road['segmentation'], (bw, bh), '%sRoad2.png' % str(patch_seq).zfill(6))

			if True:
				saveEdgeImg(road['segmentation'], (bw, bh), '%s.png' % str(patch_seq).zfill(6))
				polygons = extractPolygons(road['segmentation'])
				for pid, polygon in enumerate(polygons):
					savePolygonImg(polygon, (bw, bh), '%s_%d.png' % (str(patch_seq).zfill(6), pid))

			roads.append(road)
			saveEdgeImg(road['segmentation'], (bw, bh), './%sPatch/%sRoad.png' % (city_name, str(patch_seq).zfill(6)))
			patch_seq += 1
			ann_seq += 1

	return idx, patches, roads

def saveJSON(result, city_name):
	for sub_res, set_type in zip(result, ['Train', 'Val', 'Test']):
		with open('%sRoad%s.json' % (city_name, set_type), 'w') as outfile:
			json.dump(sub_res, outfile, cls = NumpyEncoder)
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	city_info = config.CITY_INFO[city_name]

	path = '%sPatch' % city_name
	if not os.path.exists(path):
		os.popen('mkdir %s' % path)

	p = RoadPool()
	d = np.load('%sRoadList.npy' % city_name).item()
	for rid in d:
		for lon, lat, nid in d[rid]:
			p.addV(nid, (lon, lat))
		for i in range(1, len(d[rid])):
			nid1, nid2 = d[rid][i - 1][2], d[rid][i][2]
			p.addE(nid1, nid2)
			p.addE(nid2, nid1)
	p.sortV()

	result = [{
		'info': {
			'contributor': 'Zuoyue Li',
			'about': '%s dataset for %s roads' % (set_type, city_name),
		},
		'categories': [
			{'id': 999999, 'name': 'road', 'supercategory': 'road'}
		],
		'images': [],
		'annotations': []
	} for set_type in ['training', 'validation', 'test']]

	map_info = np.load('%sMapInfo.npy' % city_name).item()
	map_list = [int(item.split('/')[-1].replace('.png', '')) for item in glob.glob('./%sMap/*' % city_name)]
	map_list.sort()

	for mid in map_list:
		print('Map ID:', mid)
		patch_seq = sum([len(item['images']) for item in result])
		ann_seq = sum([len(item['annotations']) for item in result])
		idx, patches, roads = cropMap(p, map_info, mid, city_info, patch_seq, ann_seq)
		result[idx]['images'].extend(patches)
		result[idx]['annotations'].extend(roads)
		continue
		if mid > 0 and mid % 100 == 0:
			saveJSON(result, city_name)
	saveJSON(result, city_name)




