import math, sys, time, os, io, requests, json
import numpy as np
from PIL import Image, ImageDraw
from UtilityGeography import BoundingBox
from Config import Config

def clip(subjectPolygon, clipPolygon):
	# both polygons should be clockwise/anti-clockwise
	def area(polygon):
		s = 0.0
		for i in range(len(polygon)):
			x1, y1 = polygon[i - 1]
			x2, y2 = polygon[i]
			s += (x1 * y2 - x2 * y1)
		s /= 2.0
		return s

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
	return(outputList)

def clip_in_img(subjectPolygon, w, h):
	clipPolygon = [(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)]
	subjectPolygon = [[x, -y] for x, y in subjectPolygon]
	clipPolygon = [[x, -y] for x, y in clipPolygon]
	res = clip(subjectPolygon, clipPolygon)
	return [(x, -y) for x, y in res]

class BuildingPool(object):
	def __init__(self):
		self.b = {}
		self.bbox = {}
		self.opt = ['l', 'u', 'r', 'd']

	def addBuilding(self, bid, info):
		info = np.array(info)
		assert(info.shape[0] > 2 and info.shape[1] == 2)
		if bid in self.b:
			assert(info == self.b[bid])
		else:
			self.b[bid] = info
			self.bbox[bid] = [info[:,0].min(), info[:,1].min(), info[:,0].max(), info[:,1].max()]
		return

	def sortB(self):
		self.bSorted = {}
		for i, opt in enumerate(self.opt):
			self.bSorted[opt] = [(item[i], bid) for bid, item in self.bbox.items()]
			self.bSorted[opt].sort()
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

		maxLonSet = set([bid for _, bid in self.bSorted['l'][maxLonIdx:]])
		minLatSet = set([bid for _, bid in self.bSorted['u'][:minLatIdx + 1]])
		minLonSet = set([bid for _, bid in self.bSorted['r'][:minLonIdx + 1]])
		maxLatSet = set([bid for _, bid in self.bSorted['d'][maxLatIdx:]])
		to_remove = set().union(maxLonSet).union(minLatSet).union(minLonSet).union(maxLatSet)

		return set([bid for bid in self.bbox if bid not in to_remove])

if __name__ == '__main__':
	# assert(len(sys.argv) == 2)
	config = Config()
	city_name = 'Chicago'#sys.argv[1]
	city_info = config.CITY_IMAGE[city_name]

	keys = [line.strip() for line in open('GoogleMapsAPIsKeys.txt').readlines()]
	os.popen('mkdir Building%s' % city_name)

	p = BuildingPool()
	d = np.load('BuildingList%s.npy' % city_name).item()
	for bid in d:
		p.addBuilding(bid, d[bid])
	p.sortB()

	cen_lon, cen_lat = city_info['center']
	dx, dy = city_info['step']
	x1, x2 = city_info['xrange']
	y1, y2 = city_info['yrange']
	z = city_info['zoom']
	s = city_info['scale']
	img_size = city_info['size']
	img_size_1 = img_size - 1
	img_id = 0
	pad = 20
	res = []
	bsegs = [
		((         0,          0), (         0, img_size_1)),
		((         0, img_size_1), (img_size_1, img_size_1)),
		((img_size_1, img_size_1), (img_size_1,          0)),
		((img_size_1,          0), (         0,          0)),
	]
	for x in [-12]:#range(x1, x2):
		for y in [13]:#range(y1, y2):
			print('Step', x, y)
			c_lon, c_lat = cen_lon + dx * x, cen_lat + dy * y
			dd = {
				'id'  : img_id,
				'info': {
					'lon'  : c_lon,
					'lat'  : c_lat,
					'size' : img_size,
					'zoom' : z,
					'scale': s
				},
				'v'   : [],
				'e'   : []
			}
			box = BoundingBox(c_lon, c_lat, img_size, img_size, z, s)
			minLon, maxLat = box.relativePixelToLonLat(0, 0)
			maxLon, minLat = box.relativePixelToLonLat(img_size, img_size)
			bids = p.findB(minLon, maxLon, minLat, maxLat)

			img = Image.new('P', (img_size, img_size), color = 0)
			draw = ImageDraw.Draw(img)
			for bid in bids:
				b_poly = [box.lonLatToRelativePixel(x, y) for x, y in p.b[bid]]
				b_poly = clip_in_img(b_poly, w = img_size, h = img_size)
				draw.polygon(b_poly, outline = 255)
			img.show()
			quit()

			# minLon, maxLat = box.relativePixelToLonLat(-img_size, -img_size)
			# maxLon, minLat = box.relativePixelToLonLat(2 * img_size, 2 * img_size)
			# grand_vids = g.findV(minLon, maxLon, minLat, maxLat)
			# tmp_id = {}
			# for vid in vids:
			# 	grand_vids.remove(vid)
			# 	lon, lat = g.v[vid]
			# 	xx1, yy1 = box.lonLatToRelativePixel(lon, lat)
			# 	assert(0 <= xx1 and xx1 < 600)
			# 	assert(0 <= yy1 and yy1 < 600)
			# 	tmp_id[vid] = len(dd['v'])
			# 	dd['v'].append((xx1, yy1))
			# 	for evid in g.e[vid]:
			# 		if evid not in vids:
			# 			lon, lat = g.v[evid]
			# 			xx2, yy2 = box.lonLatToRelativePixel(lon, lat)
			# 			crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
			# 			crs_res = [(int(round(item[0])), int(round(item[1]))) for item in crs_res if item is not None]
			# 			crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
			# 			assert(len(crs_res) <= 2)
			# 			if len(crs_res) == 2:
			# 				assert(crs_res[0] == crs_res[1])
			# 			if crs_res:
			# 				xx2, yy2 = crs_res[0]
			# 				assert(0 <= xx2 and xx2 < 600)
			# 				assert(0 <= yy2 and yy2 < 600)
			# 				tmp_id[evid] = len(dd['v'])
			# 				dd['v'].append(crs_res[0])
			# 				dd['e'].append((tmp_id[evid], tmp_id[vid]))
			# for vid in vids:
			# 	for evid in g.e[vid]:
			# 		if vid in tmp_id and evid in tmp_id:
			# 			dd['e'].append((tmp_id[vid], tmp_id[evid]))

			# for vid in grand_vids:
			# 	lon, lat = g.v[vid]
			# 	xx1, yy1 = box.lonLatToRelativePixel(lon, lat)
			# 	for evid in g.e[vid]:
			# 		lon, lat = g.v[evid]
			# 		xx2, yy2 = box.lonLatToRelativePixel(lon, lat)
			# 		crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
			# 		crs_res = [(int(round(item[0])), int(round(item[1]))) for item in crs_res if item is not None]
			# 		crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
			# 		crs_res = list(set(crs_res))
			# 		assert(len(crs_res) <= 2)
			# 		if len(crs_res) == 2:
			# 			id1 = len(dd['v'])
			# 			dd['v'].append(crs_res[0])
			# 			id2 = len(dd['v'])
			# 			dd['v'].append(crs_res[1])
			# 			dd['e'].extend([(id1, id2), (id2, id1)])

			# img = Image.new('P', (img_size, img_size))
			# draw = ImageDraw.Draw(img)
			# for xx, yy in dd['v']:
			# 	draw.rectangle([xx-5,yy-5,xx+5,yy+5], fill = 255)
			# for id1, id2 in dd['e']:
			# 	draw.line(dd['v'][id1] + dd['v'][id2], fill = 255, width = 5)
			# img.save('Road%s/%s' % (city_name, city_name) + '_' + str(img_id).zfill(8) + '_Road.png')
			# res.append(dd)
			while True:
				try:
					img_data = requests.get(
						'https://maps.googleapis.com/maps/api/staticmap?' 					+ \
						'maptype=%s&' 			% 'satellite' 								+ \
						'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 							+ \
						'zoom=%d&' 				% z 										+ \
						'size=%dx%d&' 			% (img_size + pad * 2, img_size + pad * 2) 	+ \
						'scale=%d&' 			% s 										+ \
						'format=%s&' 			% 'png32' 									+ \
						'key=%s' 				% keys[img_id % len(keys)] 					  \
					).content
					break
				except:
					print('Try again to get the image.')
			img = np.array(Image.open(io.BytesIO(img_data)))[pad: img_size + pad, pad: img_size + pad, ...]
			Image.fromarray(img).save('Building%s/%s' % (city_name, city_name) + '_' + str(img_id).zfill(8) + '.png')
			img_id += 1

			if img_id >= 1000 and img_id % 1000 == 0:
				with open('Building%s_%d.json' % (city_name, img_id), 'w') as outfile:
					json.dump(res, outfile)

		# with open('Road%s.json' % city_name, 'w') as outfile:
		# 	json.dump(res, outfile)

