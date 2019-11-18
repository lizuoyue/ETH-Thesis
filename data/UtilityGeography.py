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

	def lonLatToRelativePixel(self, lon, lat, int_res = True):
		px, py = lonLatToPixel(lon, lat, self.z, 'float')
		if int_res:
			return math.floor(px - self.c_px + self.c_rpx), math.floor(py - self.c_py + self.c_rpy)
		else:
			return px - self.c_px + self.c_rpx, py - self.c_py + self.c_rpy

	def relativePixelToLonLat(self, x, y):
		x = self.c_px + x - self.c_rpx
		y = self.c_py + y - self.c_rpy
		return pixelToLonLat(x, y, self.z)
