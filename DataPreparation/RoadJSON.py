import math, sys, time, os, io, requests, json
import numpy as np
from PIL import Image, ImageDraw
from UtilityGeography import BoundingBox
from Config import Config

def get_crossing(s1, s2):
	xa, ya = s1[0][0], s1[0][1]
	xb, yb = s1[1][0], s1[1][1]
	xc, yc = s2[0][0], s2[0][1]
	xd, yd = s2[1][0], s2[1][1]
	a = np.matrix([
		[xb - xa, -(xd - xc)],
		[yb - ya, -(yd - yc)]
	])
	delta = np.linalg.det(a)
	if np.fabs(delta) < 1e-6:
		return None
	c = np.matrix([
		[xc - xa, -(xd - xc)],
		[yc - ya, -(yd - yc)]
	])
	d = np.matrix([
		[xb - xa, xc - xa],
		[yb - ya, yc - ya]
	])
	lamb = np.linalg.det(c) / delta
	miu = np.linalg.det(d) / delta
	if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
		x = xc + miu * (xd - xc)
		y = yc + miu * (yd - yc)
		return (x, y)
	else:
		return None

class Graph(object):
	def __init__(self):
		self.v = {}
		self.e = {}

	def addV(self, vid, info):
		if vid in self.v:
			assert(info == self.v[vid])
		else:
			self.v[vid] = info
		return

	def addE(self, vid1, vid2):
		if vid1 in self.e:
			self.e[vid1].add(vid2)
		else:
			self.e[vid1] = set([vid2])
		return

	def sortV(self):
		self.vSorted = {}
		self.vSorted['lon'] = [(lon, vid) for vid, (lon, _) in self.v.items()]
		self.vSorted['lon'].sort()
		self.vSorted['lat'] = [(lat, vid) for vid, (_, lat) in self.v.items()]
		self.vSorted['lat'].sort()
		self.minVal = {'lon': self.vSorted['lon'][ 0][0], 'lat': self.vSorted['lat'][ 0][0]}
		self.maxVal = {'lon': self.vSorted['lon'][-1][0], 'lat': self.vSorted['lat'][-1][0]}
		return

	def _findV_GQ(self, coo_type, th):
		# return index in self.vSorted
		# find the one with min idx whose val >= th
		li = self.vSorted[coo_type]
		if th <= self.minVal[coo_type]:
			return 0
		if th > self.maxVal[coo_type]:
			return -1
		l, r = 0, len(li) - 1
		while l < r:
			mid = int(math.floor((l + r) / 2))
			if li[mid][0] >= th:
				r = mid
			else:
				l = mid + 1
		return l

	def _findV_LQ(self, coo_type, th):
		# return index in self.vSorted
		# find the one with max idx whose val <= th
		li = self.vSorted[coo_type]
		if th < self.minVal[coo_type]:
			return -1
		if th >= self.maxVal[coo_type]:
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
		minLonIdx = self._findV_GQ('lon', minLon)
		maxLonIdx = self._findV_LQ('lon', maxLon)
		minLatIdx = self._findV_GQ('lat', minLat)
		maxLatIdx = self._findV_LQ('lat', maxLat)
		assert(minLonIdx <= maxLonIdx)
		assert(minLatIdx <= maxLatIdx)
		if minLonIdx == -1 or maxLonIdx == -1 or minLatIdx == -1 or maxLatIdx == -1:
			return set([])
		set1 = set([vid for _, vid in self.vSorted['lon'][minLonIdx: maxLonIdx + 1]])
		set2 = set([vid for _, vid in self.vSorted['lat'][minLatIdx: maxLatIdx + 1]])
		return set1.intersection(set2)

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	config = Config()
	city_name = sys.argv[1]
	city_info = config.CITY_IMAGE[city_name]

	keys = [line.strip() for line in open('GoogleMapsAPIsKeys.txt').readlines()]
	os.popen('mkdir Road%s' % city_name)

	g = Graph()
	d = np.load('RoadList%s.npy' % city_name).item()
	for rid in d:
		for lon, lat, nid in d[rid]:
			g.addV(nid, (lon, lat))
		for i in range(len(d[rid]) - 1):
			nid1, nid2 = d[rid][i][2], d[rid][i + 1][2]
			g.addE(nid1, nid2)
			g.addE(nid2, nid1)
	g.sortV()

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
	for x in range(x1, x2):
		for y in range(y1, y2):
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
			vids = g.findV(minLon, maxLon, minLat, maxLat)
			minLon, maxLat = box.relativePixelToLonLat(-img_size, -img_size)
			maxLon, minLat = box.relativePixelToLonLat(2 * img_size, 2 * img_size)
			grand_vids = g.findV(minLon, maxLon, minLat, maxLat)
			tmp_id = {}
			for vid in vids:
				grand_vids.remove(vid)
				lon, lat = g.v[vid]
				xx1, yy1 = box.lonLatToRelativePixel(lon, lat)
				assert(0 <= xx1 and xx1 < 600)
				assert(0 <= yy1 and yy1 < 600)
				tmp_id[vid] = len(dd['v'])
				dd['v'].append((xx1, yy1))
				for evid in g.e[vid]:
					if evid not in vids:
						lon, lat = g.v[evid]
						xx2, yy2 = box.lonLatToRelativePixel(lon, lat)
						# print((xx1, yy1), (xx2, yy2))
						crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
						# print(crs_res)
						crs_res = [item for item in crs_res if item is not None]
						crs_res = [(int(round(xx)), int(round(yy))) for xx, yy in crs_res]
						crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
						# print(crs_res)
						assert(len(crs_res) <= 2)
						if len(crs_res) == 2:
							assert(crs_res[0] == crs_res[1])
						if crs_res:
							xx2, yy2 = crs_res[0]
							assert(0 <= xx2 and xx2 < 600)
							assert(0 <= yy2 and yy2 < 600)
							tmp_id[evid] = len(dd['v'])
							dd['v'].append(crs_res[0])
							dd['e'].append((tmp_id[evid], tmp_id[vid]))
			for vid in vids:
				for evid in g.e[vid]:
					if vid in tmp_id and evid in tmp_id:
						dd['e'].append((tmp_id[vid], tmp_id[evid]))

			for vid in grand_vids:
				lon, lat = g.v[vid]
				xx1, yy1 = box.lonLatToRelativePixel(lon, lat)
				for evid in g.e[vid]:
					lon, lat = g.v[evid]
					xx2, yy2 = box.lonLatToRelativePixel(lon, lat)
					crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
					crs_res = [item for item in crs_res if item is not None]
					crs_res = [(int(round(xx)), int(round(yy))) for xx, yy in crs_res]
					crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
					assert(len(crs_res) <= 2)
					if len(crs_res) == 2:
						id1 = len(dd['v'])
						dd['v'].append(crs_res[0])
						id2 = len(dd['v'])
						dd['v'].append(crs_res[1])
						dd['e'].extend([(id1, id2), (id2, id1)])

			img = Image.new('P', (img_size, img_size))
			draw = ImageDraw.Draw(img)
			for xx, yy in dd['v']:
				draw.rectangle([xx-5,yy-5,xx+5,yy+5], fill = 255)
			for id1, id2 in dd['e']:
				draw.line(dd['v'][id1] + dd['v'][id2], fill = 255, width = 5)
			img.save('Road%s/%s' % (city_name, city_name) + '_' + str(img_id).zfill(8) + '_Road.png')
			res.append(dd)
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
			Image.fromarray(img).save('Road%s/%s' % (city_name, city_name) + '_' + str(img_id).zfill(8) + '.png')
			img_id += 1

	with open('Road%s.json' % city_name, 'w') as outfile:
		json.dump(res, outfile)

