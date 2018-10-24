class Config(object):
	def __init__(self):
		# 
		self.AREA_SIZE      = [256, 256]
		self.AREA_SIZE_2    = [int(item /  2) for item in self.AREA_SIZE]
		self.AREA_SIZE_4    = [int(item /  4) for item in self.AREA_SIZE]
		self.AREA_SIZE_8    = [int(item /  8) for item in self.AREA_SIZE]
		self.AREA_SIZE_16   = [int(item / 16) for item in self.AREA_SIZE]
		self.AREA_SIZE_32   = [int(item / 32) for item in self.AREA_SIZE]
		self.ANCHOR_SCALE   = [16, 32, 64, 128]
		self.ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
		self.FEATURE_SHAPE  = [(64, 64), (32, 32), (16, 16), (8, 8)]
		self.FEATURE_STRIDE = [4, 8, 16, 32]
		self.PATCH_SIZE     = [224, 224]
		self.PATCH_SIZE_2   = [int(item /  2) for item in self.PATCH_SIZE]
		self.PATCH_SIZE_4   = [int(item /  4) for item in self.PATCH_SIZE]
		self.PATCH_SIZE_8   = [int(item /  8) for item in self.PATCH_SIZE]
		self.PATCH_SIZE_16  = [int(item / 16) for item in self.PATCH_SIZE]
		self.PATCH_SIZE_32  = [int(item / 32) for item in self.PATCH_SIZE]

		# 
		self.BEAM_WIDTH = 5
		self.BLUR = 0.75

		self.TABLEAU20 = [
			(174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
			(196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
		]
		# 蓝 橘 绿 红 紫
		# 棕 粉 灰 黄 青
		self.TABLEAU20_DEEP = [
			( 31, 119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40), (148, 103, 189),
			(140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34), ( 23, 190, 207),
		]

		# Learning parameters
		self.NUM_ITER = 1000001
		self.MAX_NUM_VERTICES = 30
		self.LEARNING_RATE = 3e-5
		self.LSTM_OUT_CHANNEL = [64, 32, 16]
		self.V_OUT_RES = (28, 28)
		self.AREA_TRAIN_BATCH = 4
		self.AREA_VALID_BATCH = 6
		self.AREA_TEST_BATCH = 12
		self.BUILDING_TRAIN_BATCH = 12
		self.BUILDING_VALID_BATCH = 16

		self.MIN_BBOX_AREA = 256

		self.NMS_TOP_K = 500
		self.NMS_MAX_NUM_INS = 50
		self.NMS_MIN_CONFIDENCE = 0.985
		self.NMS_IOU_THRESHOLD = 0.145

