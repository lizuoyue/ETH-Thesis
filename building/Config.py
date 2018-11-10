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

		self.TABLEAU20 = [
			(174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
			(196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
		]
		# Blue  Orange Green Red   Purple
		# Brown Pink   Grey  Yello Aoi
		self.TABLEAU20_DEEP = [
			( 31, 119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40), (148, 103, 189),
			(140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34), ( 23, 190, 207),
		]

		# Learning parameters
		self.NUM_ITER = 300001 # 200001 500001 800001
		self.MAX_NUM_VERTICES = 30
		self.LEARNING_RATE = 2e-5 # 1e-4 2e-5 4e-6
		self.LSTM_OUT_CHANNEL = [64, 32, 16]
		self.V_OUT_RES = tuple(self.PATCH_SIZE_8)
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

		self.PATH = {
			'crowdAI-leonhard': {
				'img-train': '/cluster/scratch/zoli/crowdAI/train/images',
				'img-val'  : '/cluster/scratch/zoli/crowdAI/val/images',
				'img-test' : '/cluster/scratch/zoli/crowdAI/test_images',
				'ann-train': '/cluster/scratch/zoli/crowdAI/train/annotation.json',
				'ann-val'  : '/cluster/scratch/zoli/crowdAI/val/annotation.json',
				'ann-test' : None,
				'bias'     : [77.91342018, 89.78918901, 101.50963053]
			},
			'crowdAI-dalabgpu': {
				'img-train': '/local/home/zoli/data/crowdAI/train/images',
				'img-val'  : '/local/home/zoli/data/crowdAI/val/images',
				'img-test' : '/local/home/zoli/data/crowdAI/test_images',
				'ann-train': '/local/home/zoli/data/crowdAI/train/annotation.json',
				'ann-val'  : '/local/home/zoli/data/crowdAI/val/annotation.json',
				'ann-test' : None,
				'bias'     : [77.91342018, 89.78918901, 101.50963053]
			},
			'Chicago': {
				'img-train': '/local/home/zoli/data/Chicago/ChicagoPatch',
				'img-val'  : '/local/home/zoli/data/Chicago/ChicagoPatch',
				'img-test' : '/local/home/zoli/data/Chicago/ChicagoPatch',
				'ann-train': '/local/home/zoli/data/Chicago/ChicagoBuildingTrain.json',
				'ann-val'  : '/local/home/zoli/data/Chicago/ChicagoBuildingVal.json',
				'ann-test' : '/local/home/zoli/data/Chicago/ChicagoBuildingTest.json',
				'bias'     : [84.60222819, 80.07855799, 70.79262454]
			},
			'Sunnyvale': {
				'img-train': '/local/home/zoli/data/Sunnyvale/SunnyvalePatch',
				'img-val'  : '/local/home/zoli/data/Sunnyvale/SunnyvalePatch',
				'img-test' : '/local/home/zoli/data/Sunnyvale/SunnyvalePatch',
				'ann-train': '/local/home/zoli/data/Sunnyvale/SunnyvaleBuildingTrain.json',
				'ann-val'  : '/local/home/zoli/data/Sunnyvale/SunnyvaleBuildingVal.json',
				'ann-test' : '/local/home/zoli/data/Sunnyvale/SunnyvaleBuildingTest.json',
				'bias'     : [107.52448122, 106.82606879, 98.10730461]
			},
			'Boston': {
				'img-train': '/local/home/zoli/data/Boston/BostonPatch',
				'img-val'  : '/local/home/zoli/data/Boston/BostonPatch',
				'img-test' : '/local/home/zoli/data/Boston/BostonPatch',
				'ann-train': '/local/home/zoli/data/Boston/BostonBuildingTrain.json',
				'ann-val'  : '/local/home/zoli/data/Boston/BostonBuildingVal.json',
				'ann-test' : '/local/home/zoli/data/Boston/BostonBuildingTest.json',
				'bias'     : [76.86692809, 75.58776253, 67.1285512]
			}
		}

