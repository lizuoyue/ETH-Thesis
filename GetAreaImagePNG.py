import io, os, sys
import numpy as np
import requests, math
from PIL import Image, ImageDraw
ut = __import__('Utility')
bli = __import__('GetBuildingListOSM')

# center=41.960124,-87.7976490&zoom=19&size=320x320
CITY_DICT = {
	'Chicago': {
		'left-up-center': (-87.7976490, 41.960124),
		'step': (0.000268966, 0.000200000),
		'xrange': (0, 600),
		'yrange': (0, 900),
	}
}

class AreaImageDownloader(object):

	def __init__(self, google_keys_filename, city_name):
		f = open(google_keys_filename, 'r')
		self.keys = [item.strip() for item in f.readlines()]
		f.close()
		self.city_name = city_name
		self.cons = bli.BuildingListConstructor(range_vertices = (4, 15), filename = './BuildingList-%s.npy' % city_name)
		self.building_area = {}
		self.area_building = {}
		self.zoom = 19
		self.scale = 2
		self.size = (640, 640)
		self.pad = 100
		self.city = CITY_DICT[city_name]
		self.size_g = (int(math.floor((self.size[0] + self.pad * 2) / self.scale)), int(math.floor((self.size[1] + self.pad * 2) / self.scale)))

		for k in self.cons.building:
			min_lon, max_lon = 200.0, -200.0
			min_lat, max_lat = 100.0, -100.0
			for lon, lat in self.cons.building[k]:
				min_lon = min(lon, min_lon)
				min_lat = min(lat, min_lat)
				max_lon = max(lon, max_lon)
				max_lat = max(lat, max_lat)
			c_lon = (min_lon + max_lon) / 2.0
			c_lat = (min_lat + max_lat) / 2.0
			x_idx = round((c_lon - self.city['left-up-center'][0]) / self.city['step'][0])
			y_idx = round((self.city['left-up-center'][1] - c_lat) / self.city['step'][1])
			self.building_area[k] = []
			# if x_idx >= self.city['xrange'][0] and y_idx >= self.city['yrange'][0] and x_idx < self.city['xrange'][1] and y_idx < self.city['yrange'][1]:
			for xx in range(max(0, x_idx - 2), min(self.city['xrange'][1], x_idx + 3)):
				for yy in range(max(0, y_idx - 2), min(self.city['yrange'][1], y_idx + 3)):
					self.building_area[k].append((xx, yy))

		for k in self.building_area:
			li = self.building_area[k]
			for xx, yy in li:
				if (xx, yy) in self.area_building:
					self.area_building[(xx, yy)].append(k)
				else:
					self.area_building[(xx, yy)] = [k]

		# print(self.area_building)
		return

	def norm(self, p):
		l = math.sqrt(p[0] ** 2 + p[1] ** 2)
		return (p[0] / l, p[1] / l)

	def centerRight(self, p1, p2, l):
		direction = (p1[1] - p2[1], p2[0] - p1[0])
		direction = self.norm(direction)
		x = math.floor((p1[0] + p2[0]) / 2 + l * direction[0])
		y = math.floor((p1[1] + p2[1]) / 2 + l * direction[1])
		return (x, y)

	def saveImagePolygons(self, img, polygons, area_idx):
		# Save images
		img = Image.fromarray(img)
		img.save('../%s_Area/%d_%d/img.png' % (self.city_name, area_idx[0], area_idx[1]))

		for polygon in polygons:
			mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
			img = Image.alpha_composite(img, mask)

			# Decide the order of vertices
			inner_count = 0
			for i in range(len(polygon)):
				x, y = self.centerRight(polygon[i - 1], polygon[i], 5)
				try:
					inner_count += (np.sum(mask[y, x, 1: 3]) > 1.0) # <- The pixel is not red
				except:
					inner_count += 1
			if inner_count / len(polygon) < 0.5:
				polygon.reverse()

		img.save('../%s_Area/%d_%d/merge.png' % (self.city_name, area_idx[0], area_idx[1]))

		# Save as text file to save storage
		f = open('../%s_Area/%d_%d/polygons.txt' % (self.city_name, area_idx[0], area_idx[1]), 'w')
		for polygon in polygons:
			f.write('%\n')
			for p in polygon:
				f.write('%d %d\n' % p)
		f.close()

		# Return
		return

	def saveImageBBoxes(self, img, polygons, area_idx):
		# Save images
		img = Image.fromarray(img)
		bboxes = []

		for polygon in polygons:
			l, r = 10000, -10000
			u, d = 10000, -10000
			for x, y in polygon:
				l = min(l, x)
				u = min(u, y)
				r = max(r, x)
				d = max(d, y)
			l -= 20
			r += 20
			u -= 20
			d += 20
			l = min(max(l, 0), self.size[0])
			r = min(max(r, 0), self.size[0])
			u = min(max(u, 0), self.size[1])
			d = min(max(d, 0), self.size[1])
			# mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			# draw = ImageDraw.Draw(mask)
			# draw.polygon([(l, u), (l, d), (r, d), (r, u)], fill = (255, 255, 255, 0), outline = (0, 255, 0, 128))
			# img = Image.alpha_composite(img, mask)
			if l != r and u != d:
				bboxes.append((l, u, r, d))
		if not bboxes:
			return

		img.save('../%s_Area/%d_%d/img.png' % (self.city_name, area_idx[0], area_idx[1]))
		# img.save('../%s_Area/%d_%d/merge.png' % (self.city_name, area_idx[0], area_idx[1]))

		# Save as text file to save storage
		f = open('../%s_Area/%d_%d/bboxes.txt' % (self.city_name, area_idx[0], area_idx[1]), 'w')
		for bbox in bboxes:
			f.write('%d %d %d %d\n' % bbox)
		f.close()

		# Return
		return

	def getAreaAerialImage(self, idx, area_idx):
		# Create new folder
		if not os.path.exists('../%s_Area/%d_%d' % (self.city_name, area_idx[0], area_idx[1])):
			os.makedirs('../%s_Area/%d_%d' % (self.city_name, area_idx[0], area_idx[1]))

		c_lon = self.city['left-up-center'][0] + self.city['step'][0] * area_idx[0]
		c_lat = self.city['left-up-center'][1] - self.city['step'][1] * area_idx[1]

		# Get image from Google Map API
		while True:
			try:
				data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 			+ \
					'maptype=%s&' 			% 'satellite' 						+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 					+ \
					'zoom=%d&' 				% self.zoom 						+ \
					'size=%dx%d&' 			% self.size_g						+ \
					'scale=%d&' 			% self.scale 						+ \
					'format=%s&' 			% 'png32' 							+ \
					'key=%s' 				% self.keys[idx % len(self.keys)] 	  \
				).content
				img = np.array(Image.open(io.BytesIO(data)))
				break
			except:
				print('Try again to get the image.')
				pass

		# Image large
		img = img[self.pad: img.shape[0] - self.pad, self.pad: img.shape[1] - self.pad]
		# Image.fromarray(img).show()

		# Compute polygon's vertices
		polygons = []
		bbox = ut.BoundingBox(c_lon, c_lat, self.zoom, self.scale, self.size)
		for k in self.area_building[area_idx]:
			polygon = []
			for lon, lat in self.cons.building[k]:
				px, py = bbox.lonLatToRelativePixel(lon, lat)
				if not polygon or (px, py) != polygon[-1]:
					polygon.append((px, py))
			if polygon[-1] == polygon[0]:
				polygon.pop()
			polygons.append(polygon)

		# Save and return
		# self.saveImagePolygons(img, polygons, area_idx)
		self.saveImageBBoxes(img, polygons, area_idx)
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2 or len(sys.argv) == 4)
	city_name = sys.argv[1]
	objDown = AreaImageDownloader('./GoogleMapAPIKey.txt', city_name)
	keys = [k for k in objDown.area_building]
	keys.sort()
	print('Totally %d areas.' % len(keys))
	if len(sys.argv) == 4:
		beg_idx = int(sys.argv[2])
		end_idx = int(sys.argv[3])
	else:
		beg_idx = 0
		end_idx = len(keys)
	for i in range(beg_idx, end_idx):
		objDown.getAreaAerialImage(i, keys[i])

