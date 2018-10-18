class Config(object):
	def __init__(self):
		### DO NOT MODIFY ###
		self.TILE_SIZE = 256
		self.PAD = 20
		#####################
		self.CITY_INFO = {
			'Sunnyvale': {
				'map_area': [ # (Lat, Lon)
					(37.3465, -121.9413),
					(37.3547, -121.9461),
					(37.3662, -121.9503),
					(37.3784, -121.9503),
					(37.3839, -121.9370),
					(37.4007, -121.9425),
					(37.4094, -121.9601),
					(37.4110, -121.9623),
					(37.4106, -121.9921),
					(37.4142, -122.0023),
					(37.4192, -122.0290),
					(37.4005, -122.0353),
					(37.4082, -122.0696),
					(37.4210, -122.0928),
					(37.3941, -122.1269),
					(37.3739, -122.1136),
					(37.3378, -122.0675),
					(37.3156, -122.0685),
					(37.3156, -122.0057),
					(37.3278, -122.0057),
					(37.3278, -121.9645),
					(37.3419, -121.9545)
				],
				'map_zoom': 18,
				'map_scale': 2,
				'map_size': (600, 600) # (Width 1~600, Height 1~600)

			}
		}