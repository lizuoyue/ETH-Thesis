class Config(object):
	def __init__(self):
		# 
		self.AREA_SIZE      = [256, 256]
		self.ANCHOR_SCALE   = [16, 32, 64, 128]
		self.ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
		self.FEATURE_SHAPE  = [(64, 64), (32, 32), (16, 16), (8, 8)]
		self.FEATURE_STRIDE = [4, 8, 16, 32]
		self.PATCH_SIZE     = [224, 224]

		#
		self.BEAM_WIDTH = 6
		self.BLUR = 0.75

		self.TABLEAU20 = [
			(255,   0,   0), ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),  
			( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),  
			(148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),  
			(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
			(188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229),
		]

		# Learning parameters
		self.PATH_A = '/local/lizuoyue/Areas%s'
		self.PATH_B = '../../Buildings%s.zip'
		self.NUM_ITER = 100000
		self.MAX_NUM_VERTICES = 20
		self.LEARNING_RATE = 0.0005
		self.LSTM_OUT_CHANNEL = [32, 16, 8]
		self.V_OUT_RES = (28, 28)
		self.AREA_TRAIN_BATCH = 4
		self.AREA_PRED_BATCH = 8
		self.BUILDING_TRAIN_BATCH = 12
		self.BUILDING_PRED_BATCH = 16
		self.SPLIT = 0.9

		self.COLOR_MEAN = {
			'Areas': {
				'Zurich':  [ 99.40127267, 102.85743628,  95.63071881],
				'Chicago': [ 81.11714888,  81.95831831,  74.76831702],
			},
			'Buildings': {
				'Zurich' : [104.35192842, 103.41573836,  99.22422849],
				'Chicago': [ 76.90603533,  77.37561113,  71.17612697],
			},
		}

