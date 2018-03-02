class Config(object):
	def __init__(self):
		# 
		self.TILE_SIZE = 256

		# Buildings from OpenStreetMap
		self.CITY_BUILDING = {
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

		# Areas from OpenStreetMap
		self.CITY_AREA = {
			'Zurich': {
			# Each patch is around 600 pixels in both width and height with zoom level 19
				'center': (8.5468221, 47.3768587), # ETH Zurich
				'step'  : (0.0004023093551024917, -0.0002724221013823084),
				'xrange': (-144, 144),
				'yrange': (-144, 144),
			},
			'Chicago': {
			# Each patch is around 600 pixels in both width and height with zoom level 19
				'center': (-87.7380649, 41.9289708), # Kelvyn Park
				'step'  : (0.00044697723283112595, -0.0003324594428459153),
				'xrange': (-144, 144),
				'yrange': (-144, 144),
			},
		}
		self.ZOOM = 19
		self.SCALE = 1
		self.MAX_PATCH_SIZE = 600

