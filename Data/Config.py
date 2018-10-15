class Config(object):
	def __init__(self):
		# Tile size of Google Map
		self.TILE_SIZE = 256

		# Coordinates for OpenStreetMap
		self.CITY_COO = {
			'Zurich': {
			# Each patch is around 4096 pixels in both width and height with zoom level 18
				'center': (8.5411203, 47.3772191), # Zurich HB
				'step'  : (0.021971454913330746, -0.014877879030159136),
				'xrange': (-6, 6),
				'yrange': (-6, 6),
			},
			'Chicago': {
			# Each patch is around 2048 pixels in both width and height with zoom level 18
				'center': (-87.7380649, 41.9289708), # Kelvyn Park
				'step'  : (0.01098491247405775, -0.008170523267381213),
				'xrange': (-6, 6),
				'yrange': (-6, 6),
			},
		}

		# Images for Google Static Maps API
		self.CITY_IMAGE = {
			'Zurich': {
			# Each patch is around 600 pixels in both width and height with zoom level 19
				'center': (8.5468221, 47.3768587), # ETH Zurich
				'step'  : (0.0004023093551024917, -0.0002724221013823084),
				'xrange': (-144, 144),
				'yrange': (-144, 144),
				'zoom'  : 19,
				'scale' : 1,
				'size'  : 600,
			},
			'Chicago': {
			# Each patch is around 600 pixels in both width and height with zoom level 20
				'center': (-87.7380649, 41.9289708), # Kelvyn Park
				'step'  : (0.00044697723283112595, -0.0003324594428459153),
				'xrange': (-144, 144),
				'yrange': (-144, 144),
				'zoom'  : 20,
				'scale' : 1,
				'size'  : 600,
			},
		}

if __name__ == '__main__':
	config = Config()
	city_info = config.CITY_COO['Chicago']
	lon, lat = city_info['center']
	dx, dy = city_info['step']
	x1, x2 = city_info['xrange']
	y1, y2 = city_info['yrange']
	print('left :', lon + dx * x1)
	print('up   :', lat + dy * y1)
	print('right:', lon + dx * x2)
	print('down :', lat + dy * y2)
	print(lon, lat)
	print((lon + dx * x1 + lon + dx * x2) / 2, (lat + dy * y1 + lat + dy * y2) / 2)

