class Config(object):
	def __init__(self):
		# 
		self.AREA_SIZE      = 320
		self.ANCHOR_SCALE   = [12, 24, 48, 96, 192]
		self.ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
		self.FEATURE_SHAPE  = [(160, 160), (80, 80), (40, 40), (20, 20), (10, 10)]
		self.FEATURE_STRIDE = [2, 4, 8, 16, 32]

		#
		self.BEAM_WIDTH = 5

