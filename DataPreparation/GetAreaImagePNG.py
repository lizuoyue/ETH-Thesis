import io, os, sys, glob
import numpy as np
import requests, math, random
import Config, UtilityGeography, GetBuildingListOSM
from PIL import Image, ImageDraw

config = Config.Config()

class AreaImageDownloader(object):
	def __init__(self, keys_filename, city_name):
		self.city_name = city_name
		with open(keys_filename, 'r') as f:
			self.keys = [item.strip() for item in f.readlines()]
		self.cons = GetBuildingListOSM.BuildingListConstructor(num_vertices_range = (4, 20), filename = './BuildingList%s.npy' % city_name)
		self.building_area = {}
		self.area_building = {}

		self.city_info = config.CITY_AREA[city_name]

		for bid in self.cons.getBuildingIDListSorted():
			min_lon, max_lon = 200.0, -200.0
			min_lat, max_lat = 100.0, -100.0
			for lon, lat in self.cons.building[bid]:
				min_lon = min(lon, min_lon)
				min_lat = min(lat, min_lat)
				max_lon = max(lon, max_lon)
				max_lat = max(lat, max_lat)
			c_lon = (min_lon + max_lon) / 2.0
			c_lat = (min_lat + max_lat) / 2.0
			x_idx = round((c_lon - self.city_info['center'][0]) / self.city_info['step'][0])
			y_idx = round((c_lat - self.city_info['center'][1]) / self.city_info['step'][1])
			self.building_area[bid] = []
			x1, x2 = self.city_info['xrange']
			y1, y2 = self.city_info['yrange']
			for x in range(max(x1, x_idx - 2), min(x2 - 1, x_idx + 3)):
				for y in range(max(y1, y_idx - 2), min(y2 - 1, y_idx + 3)):
					self.building_area[bid].append((x - x1, y - y1))

		for bid in self.building_area:
			for x, y in self.building_area[bid]:
				if (x, y) in self.area_building:
					self.area_building[(x, y)].append(bid)
				else:
					self.area_building[(x, y)] = [bid]

		print('Totally %d areas.' % (len(self.area_building)))
		return

	def saveImagePolygons(self, img, roadmap, polygons, area_idx):
		img = Image.fromarray(img)
		roadmap = Image.fromarray(roadmap)
		for bid, polygon in polygons:
			mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))

		img.save('../../Areas%s/%d-%d/img.png' % (self.city_name, area_idx[0], area_idx[1]))
		roadmap.save('../../Areas%s/%d-%d/roadmap.png' % (self.city_name, area_idx[0], area_idx[1]))

		if False: # <- Local test
			mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			draw = ImageDraw.Draw(mask)
			for bid, polygon in polygons:
				draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
			merge = Image.alpha_composite(img, mask)
			merge.save('../../Areas%s/%d-%d/merge-1.png' % (self.city_name, area_idx[0], area_idx[1]))
			merge = Image.alpha_composite(roadmap, mask)
			merge.save('../../Areas%s/%d-%d/merge-2.png' % (self.city_name, area_idx[0], area_idx[1]))

		# Save as text file to save storage
		with open('../../Areas%s/%d-%d/polygons.txt' % (self.city_name, area_idx[0], area_idx[1]), 'w') as f:
			for bid, polygon in polygons:
				f.write('%% %d\n' % bid)
				for v in polygon:
					f.write('%d %d\n' % v)
		return

	def getAreaAerialImage(self, seq_idx, area_idx):
		# Create new folder
		if not os.path.exists('../../Areas%s/%d-%d' % (self.city_name, area_idx[0], area_idx[1])):
			os.makedirs('../../Areas%s/%d-%d' % (self.city_name, area_idx[0], area_idx[1]))

		c_lon = self.city_info['center'][0] + self.city_info['step'][0] * (area_idx[0] + self.city_info['xrange'][0])
		c_lat = self.city_info['center'][1] + self.city_info['step'][1] * (area_idx[1] + self.city_info['yrange'][0])
		pad = int((640 - config.MAX_PATCH_SIZE) / 2)
		size = config.MAX_PATCH_SIZE + pad * 2

		# Get image from Google Map API
		google_roadmap_edge_color = '0x00ff00'
		while True:
			if True:
				data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 				+ \
					'maptype=%s&' 			% 'satellite' 							+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 						+ \
					'zoom=%d&' 				% config.ZOOM 							+ \
					'size=%dx%d&' 			% (size, size)					 		+ \
					'scale=%d&' 			% config.SCALE 							+ \
					'format=%s&' 			% 'png32' 								+ \
					'key=%s' 				% self.keys[seq_idx % len(self.keys)] 	  \
				).content
				img = np.array(Image.open(io.BytesIO(data)))
				data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 				+ \
					'maptype=%s&' 			% 'roadmap' 							+ \
					'style=feature:all|element:labels|visibility:off&' 				+ \
					'style=feature:landscape.man_made|element:geometry.stroke|'		+ \
					'color:%s&' 			% google_roadmap_edge_color 			+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 						+ \
					'zoom=%d&' 				% config.ZOOM 							+ \
					'size=%dx%d&' 			% (size, size)					 		+ \
					'scale=%d&' 			% config.SCALE 							+ \
					'format=%s&' 			% 'png32' 								+ \
					'key=%s' 				% self.keys[seq_idx % len(self.keys)] 	  \
				).content
				roadmap = np.array(Image.open(io.BytesIO(data)))
				break
			else:
				print('Try again to get the image.')

		# Image large
		img = img[pad: img.shape[0] - pad, pad: img.shape[1] - pad]
		roadmap = roadmap[pad: roadmap.shape[0] - pad, pad: roadmap.shape[1] - pad, ...]
		size -= pad * 2

		# Compute polygon's vertices
		polygons = []
		bbox = UtilityGeography.BoundingBox(c_lon, c_lat, size, size, config.ZOOM, config.SCALE)
		for bid in self.area_building[area_idx]:
			polygon = []
			for lon, lat in self.cons.building[bid]:
				px, py = bbox.lonLatToRelativePixel(lon, lat)
				if not polygon or (px, py) != polygon[-1]:
					polygon.append((px, py))
			if polygon[-1] == polygon[0]:
				polygon.pop()
			polygons.append((bid, polygon))

		# Save and return
		self.saveImagePolygons(img, roadmap, polygons, area_idx)
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2 or len(sys.argv) == 4)
	city_name = sys.argv[1]
	obj = AreaImageDownloader('./GoogleMapsAPIsKeys.txt', city_name)
	area_id = [k for k in obj.area_building]
	area_id.sort()
	if len(sys.argv) == 4:
		beg_idx = int(sys.argv[2])
		end_idx = min(int(sys.argv[3]), len(area_id))
	else:
		beg_idx = 0
		end_idx = len(area_id)
	if True:
		for i in range(beg_idx, end_idx):
			print(i)
			obj.getAreaAerialImage(i, area_id[i])
	else: # <- Local test
		filenames = glob.glob('../../Buildings%s/*/' % city_name)
		filenames = [int(item.replace('../../Buildings%s/' % city_name, '').replace('/', '')) for item in filenames]
		for i, item in enumerate(filenames):
			print(i)
			if obj.building_area[item]:
				obj.getAreaAerialImage(i, obj.building_area[item][0])



