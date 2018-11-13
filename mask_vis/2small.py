from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import json

coco = COCO('./annotation-small.json')
image_ids = coco.getImgIds(catIds = coco.getCatIds())
image_ids_set = set(image_ids)
pred_anns_file = './crowdai_val_%s.json'
for net in ['vgg16', 'maskrcnn', 'panet']:
	print(net)
	anns = json.load(open(pred_anns_file % net))
	anns = [ann for ann in anns if ann['image_id'] in image_ids_set]
	with open((pred_anns_file % net).replace('.json', '_small.json'), 'w') as f:
		f.write(json.dumps(anns))
