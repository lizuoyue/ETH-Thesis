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

class RoadPool(object):
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
			return len(li)
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
		print(self.minVal['lat'], self.maxVal['lat'])
		print(minLat, maxLat)
		print(minLatIdx, maxLatIdx)
		assert(minLatIdx <= maxLatIdx)
		if minLonIdx == -1 or maxLonIdx == -1 or minLatIdx == -1 or maxLatIdx == -1:
			return set([])
		set1 = set([vid for _, vid in self.vSorted['lon'][minLonIdx: maxLonIdx + 1]])
		set2 = set([vid for _, vid in self.vSorted['lat'][minLatIdx: maxLatIdx + 1]])
		return set1.intersection(set2)

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
	bsegs = [
		(( 0,  0), ( 0, bh - 1)),
		(( 0, bh - 1), (bw - 1, bh - 1)),
		((bw - 1, bh - 1), (bw - 1,  0)),
		((bw - 1,  0), ( 0,  0)),
	]

	patches, roads = [], []
	map_img = np.array(Image.open('./%sMap/%s.png' % (city_name, str(mid).zfill(6))))
	map_box = BoundingBox(c_lon, c_lat, (w - config.PAD * 2) * s, (h - config.PAD * 2) * s, z, s)
	for x in range(x1, x2):
		for y in range(y1, y2):
			l, u = map_box.c_rpx + x * dx - int(bw / 2), map_box.c_rpy + y * dy - int(bh / 2)
			r, d = l + bw, u + bh
			Image.fromarray(map_img[u: d, l: r, ...]).save('./%sRoad/%s.png' % (city_name, str(patch_seq).zfill(6)))
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
			road['id'] = ann_seq,
			road['image_id'] = patch_seq
			road['area'] = 0
			road['bbox'] = [0, 0, 0, 0]
			road['segmentation'] = {'v': [], 'e': []}

			vids = road_pool.findV(minLon, maxLon, minLat, maxLat)
			minLon, maxLat = tmp_box.relativePixelToLonLat(l - bw, u - bh)
			maxLon, minLat = tmp_box.relativePixelToLonLat(r + bw, d + bh)
			grand_vids = road_pool.findV(minLon, maxLon, minLat, maxLat)
			tmp_id = {}
			for vid in vids:
				grand_vids.remove(vid)
				lon, lat = road_pool.v[vid]
				xx1, yy1 = tmp_box.lonLatToRelativePixel(lon, lat)
				assert(0 <= xx1 and xx1 < bw)
				assert(0 <= yy1 and yy1 < bh)
				tmp_id[vid] = len(road['segmentation']['v'])
				road['segmentation']['v'].append((xx1, yy1))
				for evid in road_pool.e[vid]:
					if evid not in vids:
						lon, lat = road_pool.v[evid]
						xx2, yy2 = tmp_box.lonLatToRelativePixel(lon, lat)
						crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
						crs_res = [(int(round(item[0])), int(round(item[1]))) for item in crs_res if item is not None]
						crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
						assert(len(crs_res) <= 2)
						if len(crs_res) == 2:
							assert(crs_res[0] == crs_res[1])
						if crs_res:
							xx2, yy2 = crs_res[0]
							assert(0 <= xx2 and xx2 < bw)
							assert(0 <= yy2 and yy2 < bh)
							tmp_id[evid] = len(road['segmentation']['v'])
							road['segmentation']['v'].append(crs_res[0])
							road['segmentation']['e'].append((tmp_id[evid], tmp_id[vid]))
			for vid in vids:
				for evid in road_pool.e[vid]:
					if vid in tmp_id and evid in tmp_id:
						road['segmentation']['e'].append((tmp_id[vid], tmp_id[evid]))

			for vid in grand_vids:
				lon, lat = road_pool.v[vid]
				xx1, yy1 = tmp_box.lonLatToRelativePixel(lon, lat)
				for evid in road_pool.e[vid]:
					lon, lat = road_pool.v[evid]
					xx2, yy2 = tmp_box.lonLatToRelativePixel(lon, lat)
					crs_res = [get_crossing(((xx1, yy1), (xx2, yy2)), bseg) for bseg in bsegs]
					crs_res = [(int(round(item[0])), int(round(item[1]))) for item in crs_res if item is not None]
					crs_res = [item for item in crs_res if item != (xx1, yy1) and item != (xx2, yy2)]
					crs_res = list(set(crs_res))
					assert(len(crs_res) <= 2)
					if len(crs_res) == 2:
						id1 = len(road['segmentation']['v'])
						road['segmentation']['v'].append(crs_res[0])
						id2 = len(road['segmentation']['v'])
						road['segmentation']['v'].append(crs_res[1])
						road['segmentation']['e'].extend([(id1, id2), (id2, id1)])
			roads.append(road)

			ann_img = Image.new('P', (bw, bh), color = 255)
			draw = ImageDraw.Draw(ann_img)
			for xx, yy in road['segmentation']['v']:
				draw.rectangle([xx-5,yy-5,xx+5,yy+5], fill = 0)
			for id1, id2 in road['segmentation']['e']:
				draw.line(road['segmentation']['v'][id1] + road['segmentation']['v'][id2], fill = 0, width = 5)
			ann_img.save('./%sRoad/%sAnn.png' % (city_name, str(patch_seq).zfill(6)))

			patch_seq += 1
			ann_seq += 1

	return idx, patches, roads

def saveJSON(result, city_name):
	for sub_res, set_type in zip(result, ['Train', 'Val', 'Test']):
		with open('%sRoad%s.json' % (city_name, set_type), 'w') as outfile:
			json.dump(sub_res, outfile, cls = NumpyEncoder)
	return

if __name__ == '__main__':
	# assert(len(sys.argv) == 2)
	city_name = 'Chicago'#sys.argv[1]
	city_info = config.CITY_INFO[city_name]

	path = '%sRoad' % city_name
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
		if mid >= 0 and mid % 100 == 0:
			saveJSON(result, city_name)
	saveJSON(result, city_name)




