from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os

coco = COCO('SunnyvaleBuildingTrain.json')
category_ids = coco.loadCats(coco.getCatIds())
print(category_ids)
image_ids = coco.getImgIds(catIds=coco.getCatIds())

for _ in range(100):
	random_image_id = random.choice(image_ids)
	print(random_image_id)
	img = coco.loadImgs(random_image_id)[0]
	print(img)

	image_path = os.path.join('SunnyvaleBuilding', img["file_name"])
	I = io.imread(image_path)

	annotation_ids = coco.getAnnIds(imgIds=img['id'])
	annotations = coco.loadAnns(annotation_ids)

	print(annotations)

	plt.imshow(I)
	plt.axis('off')
	coco.showAnns(annotations)
	plt.show()
