class Config(object):
	def __init__(self):
		### DO NOT MODIFY ###
		self.TILE_SIZE = 256
		self.PAD = 20
		self.OSM_LON_STEP = 0.02
		self.OSM_LAT_STEP = 0.01
		#####################
		self.CITY_INFO = {
			'Chicago': {
				'city_name': 'Chicago',
				'map_area': [ # (Lat, Lon)
					(41.9800, -87.7300),
					(41.7500, -87.7300),
					(41.7500, -87.5600),
					(41.8000, -87.6600),
					(41.9800, -87.6600)
				],
				'map_zoom': 18,
				'map_scale': 2,
				'map_size': (600, 600), # (Width 1~600, Height 1~600)
				'b_size': (300, 300),
				'b_step': (150, 150),
				'b_x_range': (-2, 2),
				'b_y_range': (-2, 2),
				'r_size': (300, 300),
				'r_step': (150, 150),
				'r_x_range': (-2, 2),
				'r_y_range': (-2, 2),
				'val_test': lambda lon, lat: 0 if lat >= 41.7680 else (1 if lon <= -87.6480 else 2) # 0,1,2 - train,val,test
			},
			'Sunnyvale': {
				'city_name': 'Sunnyvale',
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
				'map_size': (600, 600), # (Width 1~600, Height 1~600)
				'b_size': (300, 300),
				'b_step': (150, 150),
				'b_x_range': (-2, 2),
				'b_y_range': (-2, 2),
				'r_size': (300, 300),
				'r_step': (150, 150),
				'r_x_range': (-2, 2),
				'r_y_range': (-2, 2),
				'val_test': lambda lon, lat: 0 if lon >= -122.1000 else (1 if lat <= 37.3920 else 2) # 0,1,2 - train,val,test
			},
			'Boston': {
				'city_name': 'Boston',
				'map_area': [ # (Lat, Lon)
					(42.3535, -71.1100),
					(42.3935, -71.0000),
					(42.4335, -71.0000),
					(42.4335, -71.1840),
					(42.3085, -71.1840),
					(42.3085, -71.0550),
					(42.3400, -71.0450)
				],
				'map_zoom': 18,
				'map_scale': 2,
				'map_size': (600, 600), # (Width 1~600, Height 1~600)
				'b_size': (300, 300),
				'b_step': (150, 150),
				'b_x_range': (-2, 2),
				'b_y_range': (-2, 2),
				'r_size': (300, 300),
				'r_step': (150, 150),
				'r_x_range': (-2, 2),
				'r_y_range': (-2, 2),
				'val_test': lambda lon, lat: 0 if lon <= -71.0300 else (1 if lat <= 42.4150 else 2) # 0,1,2 - train,val,test
			}
		}
		self.OSM_HIGHWAY_BLACKLIST = {
			'pedestrian', 'footway', 'bridleway', 'steps', 'path', 'sidewalk', 'cycleway', 'proposed',
			'construction', 'bus_stop', 'crossing', 'elevator', 'emergency_access_point', 'escape', 'give_way',
			'motorway', 'trunk'
		}



