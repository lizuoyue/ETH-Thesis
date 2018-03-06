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

def lonLatToPixel(lon, lat, z):
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	return math.floor(wx * scale), math.floor(wy * scale)

def lonLatToTile(lon, lat, z):
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	return math.floor(wx * scale / config.TILE_SIZE), math.floor(wy * scale / config.TILE_SIZE)

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
		self.width, self.height = width, height
		self.z = zoom + scale - 1
		self.c_px, self.c_py = lonLatToPixel(c_lon, c_lat, self.z)

	def lonLatToRelativePixel(self, lon, lat):
		px, py = lonLatToPixel(lon, lat, self.z)
		return math.floor(px - self.c_px + self.width / 2), math.floor(py - self.c_py + self.height / 2)


if __name__ == '__main__':
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

