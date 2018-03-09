import io, os, sys
import numpy as np
import requests, math, random
import Config, UtilityGeography, GetBuildingListOSM
from PIL import Image, ImageDraw

config = Config.Config()

class BuildingImageDownloader(object):
	def __init__(self, keys_filename, city_name):
		self.city_name = city_name
		with open(keys_filename, 'r') as f:
			self.keys = [item.strip() for item in f.readlines()]

	def saveImagePolygon(self, img, polygon, roadmap):
		# Save images
		img = Image.fromarray(img)
		roadmap = Image.fromarray(roadmap)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		img.save('../../Buildings%s/%d/img.png' % (self.city_name, building_id))
		roadmap.save('../../Buildings%s/%d/roadmap.png' % (self.city_name, building_id))
		if False:
			mask.save('../../Buildings%s/%d/mask.png' % (self.city_name, building_id))
			merge = Image.alpha_composite(img, mask)
			merge.save('../../Buildings%s/%d/merge-1.png' % (self.city_name, building_id))
			merge = Image.alpha_composite(roadmap, mask)
			merge.save('../../Buildings%s/%d/merge-2.png' % (self.city_name, building_id))

		# Decide the order of vertices
		s = sum([x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1: ] + [polygon[0]])])
		if s < 0:
			polygon.reverse()

		# Save as text file to save storage
		with open('../../Buildings%s/%d/polygon.txt' % (self.city_name, building_id), 'w') as f:
			for vertex in polygon:
				f.write('%d %d\n' % vertex)
		return

	def getBuildingAerialImage(self, seq_idx, building_id, building):
		# Decide the tight bounding box
		min_lon, max_lon = 200.0, -200.0
		min_lat, max_lat = 100.0, -100.0
		for lon, lat in building:
			min_lon = min(lon, min_lon)
			min_lat = min(lat, min_lat)
			max_lon = max(lon, max_lon)
			max_lat = max(lat, max_lat)
		c_lon = (min_lon + max_lon) / 2.0
		c_lat = (min_lat + max_lat) / 2.0

		# Decide the zoom level
		left, up = UtilityGeography.lonLatToPixel(min_lon, max_lat, config.ZOOM)
		right, down = UtilityGeography.lonLatToPixel(max_lon, min_lat, config.ZOOM)
		diameter = math.sqrt((right - left) ** 2 + (down - up) ** 2)
		patch_size = math.ceil(1.1 * diameter) # <- Pad
		if patch_size > config.MAX_PATCH_SIZE:
			return
		pad = int((640 - config.MAX_PATCH_SIZE) / 2)
		size = patch_size + pad * 2

		# Create new folder
		if not os.path.exists('../../Buildings%s/%d' % (self.city_name, building_id)):
			os.makedirs('../../Buildings%s/%d' % (self.city_name, building_id))

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
		img = img[pad: img.shape[0] - pad, pad: img.shape[1] - pad, ...]
		roadmap = roadmap[pad: roadmap.shape[0] - pad, pad: roadmap.shape[1] - pad, ...]

		# Compute polygon's vertices
		size -= pad * 2
		bbox = UtilityGeography.BoundingBox(c_lon, c_lat, size, size, config.ZOOM, config.SCALE)
		polygon = [] # <- polygon: (col, row)
		for lon, lat in building:
			px, py = bbox.lonLatToRelativePixel(lon, lat)
			if not polygon or (px, py) != polygon[-1]:
				polygon.append((px, py))
		if polygon[-1] == polygon[0]:
			polygon.pop()

		# Save and return
		self.saveImagePolygon(img, polygon, roadmap)
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 4)
	city_name, idx_beg, idx_end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
	objCons = GetBuildingListOSM.BuildingListConstructor((4, 20), './BuildingList-%s.npy' % city_name)
	objDown = BuildingImageDownloader('./GoogleMapsAPIsKeys.txt', city_name)
	id_list = objCons.getBuildingIDListSorted()
	for i, building_id in enumerate(id_list):
		if i < idx_beg or i >= idx_end:
			continue
		print(i)
		objDown.getBuildingAerialImage(i, building_id, objCons.getBuilding(building_id))

