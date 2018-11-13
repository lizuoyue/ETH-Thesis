from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os, sys
import json, colorsys
from skimage.measure import find_contours
from matplotlib import patches, lines
from matplotlib.patches import Polygon, Ellipse

figsize = (8, 8)

TABLEAU20 = [
	(174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
	(196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
# Blue  Orange Green Red   Purple
# Brown Pink   Grey  Yello Aoi
TABLEAU20_DEEP = [
	( 31, 119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40), (148, 103, 189),
	(140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34), ( 23, 190, 207),
]

def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + alpha * color[c] * 255,
								  image[:, :, c])
	return image

def random_colors(N):
	return [(tuple(np.array(TABLEAU20[i%10])/255.0), tuple(np.array(TABLEAU20_DEEP[i%10])/255.0)) for i in range(N)]
	# return (np.random.random((N, 3))*0.6+0.4)

def display_instances(image, boxes, masks, filename, figsize = figsize):
	"""
	boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
	masks: [height, width, num_instances]
	figsize: (optional) the size of the image.
	"""
	# Number of instances
	N = boxes.shape[0]
	assert(N == masks.shape[-1])

	_, ax = plt.subplots(1, figsize = figsize)

	# Generate random colors
	colors = random_colors(N)

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	# ax.set_ylim(height + 10, -10)
	# ax.set_xlim(-10, width + 10)
	ax.axis('off')

	masked_image = image.astype(np.uint32).copy()
	for i in range(N):
		c1, c2 = colors[i]

		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
							  alpha=0.5, #linestyle="dashed",
							  edgecolor=(0,1,0,0.5), facecolor='none')
		ax.add_patch(p)

		# Mask
		mask = masks[:, :, i]
		# masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor=c1+(0.5,), edgecolor=c2+(0.75,), linewidth=2, joinstyle='round')
			ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	plt.savefig(filename)




if __name__ == '__main__':
	np.random.seed(8888)
	os.popen('mkdir new_res')

	IMAGES_DIRECTORY = './images'
	ANNOTATIONS_PATH = './annotation-small.json'
	VAL_ANNOTATIONS_PATH = './crowdai_val_%s_small.json'

	coco = COCO(ANNOTATIONS_PATH)
	image_ids = coco.getImgIds(catIds = coco.getCatIds())
	# image_ids = list(np.random.choice(image_ids, 100, replace = False))
	image_ids = [9843, 16502, 18848, 33232, 38410, 39412, 41635, 54737, 59830]

	choose = ['vgg16']#, 'panet', 'maskrcnn']
	d = {}
	for c in choose:
		d[c] = {img_id: [] for img_id in image_ids}
		pred_anns = json.load(open(VAL_ANNOTATIONS_PATH % c))
		for pred_ann in pred_anns:
			if pred_ann['image_id'] in d[c]:
				pred_ann['iscrowd'] = 0
				if c != 'vgg16':
					pred_ann['segmentation'] = cocomask.decode(pred_ann['segmentation'])
				d[c][pred_ann['image_id']].append(pred_ann)

	for img_id in image_ids:
		print(img_id)
		for c in choose:
			img = coco.loadImgs([img_id])[0]
			image_path = os.path.join(IMAGES_DIRECTORY, img['file_name'])
			image = io.imread(image_path)

			anns = d[c][img_id]
			if c != 'vgg16':
				boxes, masks = [], []
				for ann in anns:
					x1,y1,w,h=ann['bbox']
					boxes.append([y1,x1,y1+h,x1+w])
					masks.append(ann['segmentation'][..., np.newaxis])
				if boxes:
					boxes = np.array(boxes)
					masks = np.concatenate(masks, axis = -1)
				else:
					boxes = np.zeros((0, 4))
					masks = np.zeros((300, 300, 0))
				display_instances(image, boxes, masks, './res/%d_%s.pdf' % (img_id, c))
			else:
				_, ax = plt.subplots(1, figsize = figsize)
				ax.imshow(image)
				ax.axis('off')
				colors = random_colors(len(anns))
				for (c1, c2), ann in zip(colors, anns):
					x1,y1,w,h=ann['bbox']
					p = patches.Rectangle((x1, y1), w, h, linewidth=1,
										  alpha=0.5,
										  edgecolor=(0,1,0,0.5), facecolor='none')
					ax.add_patch(p)
					polygon = np.array([ann['segmentation'][0][0::2], ann['segmentation'][0][1::2]]).transpose()
					
					p = Polygon(polygon, facecolor=c1+(0.5,), edgecolor=c2+(0.75,), linewidth=2, joinstyle='round')
					ax.add_patch(p)

					for i in range(polygon.shape[0]):
						p = Ellipse(polygon[i], width = 4, height = 4, facecolor=c2+(0.75,), edgecolor=c2+(0.75,), linewidth=2)
						ax.add_patch(p)
				plt.savefig('./new_res/%d_%s.pdf' % (img_id, c))




