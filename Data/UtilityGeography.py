import math
import Config

config = Config.Config()

def lonLatToWorld(lon, lat):
	# Truncating to 0.9999 effectively limits latitude to 89.189.
	# This is about a third of a tile past the edge of the world tile.
	siny = math.sin(float(lat) * math.pi / 180.0);
	siny = min(max(siny, -0.9999), 0.9999)
	return (
		config.TILE_SIZE * (0.5 + float(lon) / 360.0),
		config.TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
	)

def lonLatToPixel(lon, lat, z, mode = 'int'):
	assert(mode in ['int', 'float'])
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	if mode == 'int':
		res = (math.floor(wx * scale), math.floor(wy * scale))
	else:
		res = (wx * scale, wy * scale)
	return res

def lonLatToTile(lon, lat, z, mode = 'float'):
	assert(mode in ['int', 'float'])
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	if mode == 'int':
		res = (math.floor(wx * scale / config.TILE_SIZE), math.floor(wy * scale / config.TILE_SIZE))
	else:
		res = (wx * scale / config.TILE_SIZE, wy * scale / config.TILE_SIZE)
	return res

def pixelToLonLat(px, py, z):
	# Return the longitude of the left boundary of a pixel,
	# and the latitude of the upper boundary of a pixel.
	scale = 2 ** z
	lon = (float(px) / float(scale) / config.TILE_SIZE - 0.5) * 360.0
	temp = math.exp((0.5 - float(py) / float(scale) / config.TILE_SIZE) * 4 * math.pi)
	lat = math.asin((temp - 1.0) / (temp + 1.0)) / math.pi * 180.0
	return lon, lat

class BoundingBox(object):
	def __init__(self, c_lon, c_lat, width, height, zoom, scale):
		self.w, self.h = width, height
		self.z = zoom + scale - 1
		self.c_rpx, self.c_rpy = math.floor(width / 2), math.floor(height / 2)
		self.c_px, self.c_py = lonLatToPixel(c_lon, c_lat, self.z)
		self.tc_lon, self.tc_lat = self.relativePixelToLonLat(self.c_rpx, self.c_rpy)

	def lonLatToRelativePixel(self, lon, lat):
		px, py = lonLatToPixel(lon, lat, self.z, 'float')
		return math.floor(px - self.c_px + self.c_rpx), math.floor(py - self.c_py + self.c_rpy)

	def relativePixelToLonLat(self, x, y):
		x = math.floor(self.c_px + x - self.c_rpx)
		y = math.floor(self.c_py + y - self.c_rpy)
		return pixelToLonLat(x, y, self.z)


if __name__ == '__main__':
	c_lon, c_lat = -122.04153, 37.36129
	bbox = BoundingBox(c_lon, c_lat, 1200, 1200, 18, 2)
	print(bbox.tc_lon, bbox.tc_lat)
	r, d = bbox.relativePixelToLonLat(1200, 1200)
	print(r, d)

	import io, requests
	from PIL import Image
	import numpy as np
	imgs = []
	for c_lon, c_lat in [(bbox.tc_lon, bbox.tc_lat), (r, d)]:
		while True:
			try:
				img_data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 					+ \
					'maptype=%s&' 			% 'satellite' 								+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 							+ \
					'zoom=%d&' 				% 18 										+ \
					'size=%dx%d&' 			% (640, 640) 								+ \
					'scale=%d&' 			% 2 										+ \
					'format=%s&' 			% 'png32' 									+ \
					'key=%s' 				% 'AIzaSyCyAEx-G-TOsUEq6me1nOpNhsA7OoROQWw'   \
				).content
				break
			except:
				print('Try again to get the image.')
		pad, img_size = 40, 1200
		img = Image.open(io.BytesIO(img_data))
		img = np.array(img)[pad: img_size + pad, pad: img_size + pad, ...]
		imgs.append(img)
	Image.fromarray(imgs[0][600:, 600:, ...]).show()
	Image.fromarray(imgs[1][:600, :600, ...]).show()

	quit()
	zoom = 18
	patch_size = 2048
	city_info = config.CITY_BUILDING['Chicago']
	lon, lat = city_info['center']
	dx, dy = city_info['step']
	x_1, x_2 = city_info['xrange']
	y_1, y_2 = city_info['yrange']
	if False:
		px1, py1 = lonLatToPixel(lon, lat, zoom)
		lon_n, lat_n = pixelToLonLat(px1 + patch_size, py1 + patch_size, zoom)
		print(lon_n - lon, lat_n - lat)
	else:
		for x in range(x_1, x_2):
			for y in range(y_1, y_2):
				if True:
					print('%lf,%lf' % (lat + dy * y, lon + dx * x))
				else:
					px1, py1 = lonLatToPixel(lon + dx * x, lat + dy * y, zoom)
					px2, py2 = lonLatToPixel(lon + dx * x + dx, lat + dy * y + dy, zoom)
					print(x, y, px2 - px1, py2 - py1)


