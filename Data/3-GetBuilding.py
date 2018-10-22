import math, sys, time, os, io, requests, json, glob
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

class BuildingPool(object):
	def __init__(self):
		self.b = {}
		self.bbox = {}
		self.opt = ['l', 'u', 'r', 'd']

	def addBuilding(self, bid, info):
		info = np.array(info)
		if not (info.shape[0] > 2 and info.shape[1] == 2):
			return
		if bid in self.b:
			assert(info == self.b[bid])
		else:
			self.b[bid] = info
			self.bbox[bid] = [info[:,0].min(), info[:,1].max(), info[:,0].max(), info[:,1].min()]
		return

	def sortB(self):
		self.bidSet = set(self.bbox.keys())
		self.bSorted = {}
		self.bSortedID = {}
		for i, opt in enumerate(self.opt):
			self.bSorted[opt] = [(item[i], bid) for bid, item in self.bbox.items()]
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

	def findB(self, minLon, maxLon, minLat, maxLat):
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

def cropMap(building_pool, map_info, mid, city_info, patch_seq, ann_seq):
	city_name = city_info['city_name']
	dx, dy = city_info['b_step']
	bw, bh = city_info['b_size']
	x1, x2 = city_info['b_x_range']
	y1, y2 = city_info['b_y_range']

	info = map_info[mid]
	c_lat, c_lon = info['center']
	w, h = info['size']
	z, s = info['zoom'], info['scale']
	idx = city_info['val_test'](c_lon, c_lat)

	patches, buildings = [], []
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

			ann_img = Image.new('P', (bw, bh), color = 0)
			draw = ImageDraw.Draw(ann_img)
			bids = building_pool.findB(minLon, maxLon, minLat, maxLat)
			for bid in bids:
				if len(p.b[bid]) >= 3:
					b_poly = [tmp_box.lonLatToRelativePixel(x, y, int_res = False) for x, y in p.b[bid]]
					b_poly = clip_in_img(b_poly, w = bw, h = bh)
					a = -area(b_poly)
					if len(b_poly) >= 3 and a >= 1.0:
						draw.polygon(b_poly, outline = 255)
						tmp_poly = np.array(b_poly)
						building = {'category_id': 100, 'iscrowd': 0}
						building['id'] = ann_seq
						building['image_id'] = patch_seq
						building['segmentation'] = [list(tmp_poly.reshape(len(b_poly) * 2))]
						building['area'] = a
						l, u, r, d = tmp_poly[:, 0].min(), tmp_poly[:, 1].min(), tmp_poly[:, 0].max() + 1, tmp_poly[:, 1].max() + 1
						building['bbox'] = [l, u, r - l, d - u]
						building['building_id'] = bid
						buildings.append(building)
						ann_seq += 1

			ann_img.save('./%sPatch/%sBuilding.png' % (city_name, str(patch_seq).zfill(6)))
			patch_seq += 1

	return idx, patches, buildings

def saveJSON(result, city_name):
	for sub_res, set_type in zip(result, ['Train', 'Val', 'Test']):
		with open('%sBuilding%s.json' % (city_name, set_type), 'w') as outfile:
			json.dump(sub_res, outfile, cls = NumpyEncoder)
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	city_info = config.CITY_INFO[city_name]

	path = '%sPatch' % city_name
	if not os.path.exists(path):
		os.popen('mkdir %s' % path)

	p = BuildingPool()
	d = np.load('%sBuildingList.npy' % city_name).item()
	for bid in d:
		p.addBuilding(bid, d[bid])
	p.sortB()

	result = [{
		'info': {
			'contributor': 'Zuoyue Li',
			'about': '%s dataset for %s buildings' % (set_type, city_name),
		},
		'categories': [
			{'id': 100, 'name': 'building', 'supercategory': 'building'}
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
		idx, patches, buildings = cropMap(p, map_info, mid, city_info, patch_seq, ann_seq)
		result[idx]['images'].extend(patches)
		result[idx]['annotations'].extend(buildings)
		if mid >= 0 and mid % 100 == 0:
			saveJSON(result, city_name)
	saveJSON(result, city_name)




