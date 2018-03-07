class Config(object):
	def __init__(self):
		# 
		self.AREA_SIZE      = [320, 320]
		self.TRAIN_AREA_BATCH = 4
		self.PRED_AREA_BATCH = 8
		self.ANCHOR_SCALE   = [12, 24, 48, 96, 192]
		self.ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
		self.FEATURE_SHAPE  = [(160, 160), (80, 80), (40, 40), (20, 20), (10, 10)]
		self.FEATURE_STRIDE = [2, 4, 8, 16, 32]
		self.PATCH_SIZE     = [256, 256]

		#
		self.BEAM_WIDTH = 5
		self.BLUR = 0.75

		self.TABLEAU20 = [
			( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),  
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
		self.V_OUT_RES = (32, 32)
		self.AREA_TRAIN_BATCH = 4
		self.AREA_PRED_BATCH = 8
		self.BUILDING_TRAIN_BATCH = 12