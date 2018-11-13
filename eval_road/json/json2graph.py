import json, os, time
import numpy as np
from scipy import spatial

city = 'Sunnyvale'

def l2dist(a, b):
	diff = np.array(a) - np.array(b)
	return np.sqrt(np.dot(diff, diff))

coco = json.load(open('%sRoadVal.json' % city))

li = [(item['box_lon_lat'][0], -item['box_lon_lat'][1], item) for item in coco['images']]
li = [item for item in li if item[2]['map_crop_lurd'][0] % 100 == 50 and item[2]['map_crop_lurd'][1] % 100 == 50]
li.sort()
li = [item[2] for item in li]

lons, lats = set(), set()
for item in li:
	lon, lat = item['box_lon_lat'][0:2]
	lons.add(lon)
	lats.add(lat)
lons = sorted(list(lons))
lats = sorted(list(lats), reverse = True)
lon2bx = {v: k for k, v in enumerate(lons)}
lat2by = {v: k for k, v in enumerate(lats)}

bias_by_img_id = {}
for item in li:
	lon, lat = item['box_lon_lat'][0:2]
	bias_by_img_id[item['id']] = (lon2bx[lon], lat2by[lat])

anns = [ann for ann in coco['annotations'] if ann['image_id'] in bias_by_img_id]
e_val = []
for ann in anns:
	bx, by = bias_by_img_id[ann['image_id']]
	for v1, v2 in ann['segmentation']:
		e_val.append((
			(v1[0] + bx * 300, v1[1] + by * 300),
			(v2[0] + bx * 300, v2[1] + by * 300)
		))

v_val_set = set()
for v1, v2 in e_val:
	v_val_set.add(v1)
	v_val_set.add(v2)
v_val_li = sorted(list(v_val_set))
v_val2idx = {v: k for k, v in enumerate(v_val_li)}
e_idx = [[v_val2idx[v1], v_val2idx[v2]] for v1, v2 in e_val]

kd_tree = spatial.KDTree(v_val_li)
pairs = kd_tree.query_pairs(r = 2)
e_idx.extend(list(pairs))

with open('%s.graph' % city, 'w') as f:
	for x, y in v_val_li:
		f.write('%d %d\n' % (x, y))
	f.write('\n')
	for s, t in e_idx:
		if s != t:
			f.write('%d %d\n' % (s, t))

from PIL import Image, ImageDraw
img = Image.new('RGB', (4096, 4096), color = (0, 0, 0))
draw = ImageDraw.Draw(img)
for s, t in e_idx:
	draw.line(v_val_li[s] + v_val_li[t], width = 5, fill = (255, 255, 255))
for x, y in v_val_li:
	draw.rectangle([x-5,y-5,x+5,y+5], fill = (255, 0, 0))
img.show()


