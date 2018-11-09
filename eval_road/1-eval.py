import json, os, time
import numpy as np

def l2dist(a, b):
	diff = np.array(a) - np.array(b)
	return np.sqrt(np.dot(diff, diff))

search_r = 3
search_range = []
for i in range(-search_r, search_r + 1):
	for j in range(-search_r, search_r + 1):
		if l2dist((0, 0), (i, j)) <= search_r:
			search_range.append((i, j))

def near(vd, v):
	x, y = v
	for i, j in search_range:
		if (x + i, y + j) in vd:
			return vd.pop((x + i, y + j))
	return None

sel = 'big' # 'dalabgpu'
sel_size = 512 # 256
sel_num = round(4096 / sel_size)

dts = json.load(open('../../EvalR/predictions_roadtracer-%s_vgg16_test.json' % sel))
gts = json.load(open('../../EvalR/roadtracer_test_big.json'))
gt_img_by_id = {}
for gt in gts['images']:
	gt_img_by_id[gt['id']] = gt

res = {}
for dt in dts:
	img_info = gt_img_by_id[dt['image_id']]
	lurd = img_info['tile_crop_lurd']
	# if not (lurd[0] % sel_size == 0 and lurd[1] % sel_size == 0):
	# 	continue
	x_bias, y_bias = img_info['tile_bias']
	city = img_info['tile_file'].split('_')[0]
	if city not in res:
		res[city] = {'v': [], 'vd': {}, 'e': []}
	assert(len(res[city]['v']) >= len(res[city]['vd']))
	base_idx = len(res[city]['v'])
	vs = [(item[0] + round(lurd[0] / sel_size * 28) + x_bias * sel_num * 28, item[1] + round(lurd[1] / sel_size * 28) + y_bias * sel_num * 28) for item in dt['vertices']]
	r_idx2a_idx = {}
	for r_idx, v in enumerate(vs):
		f = near(res[city]['vd'], v)
		if f is None:
			r_idx2a_idx[r_idx] = base_idx
			res[city]['v'].append(v)
			res[city]['vd'][v] = (base_idx, 1)
			base_idx += 1
		else:
			fv, fn = f
			old_v = res[city]['v'][fv]
			new_v = (round((old_v[0] * fn + v[0]) / (fn + 1)), round((old_v[1] * fn + v[1]) / (fn + 1)))
			if new_v not in res[city]['vd']:
				r_idx2a_idx[r_idx] = fv
				res[city]['v'][fv] = new_v
				res[city]['vd'][new_v] = (fv, fn + 1)
			else:
				tp = res[city]['vd'][new_v]
				r_idx2a_idx[r_idx] = tp[0]
				res[city]['vd'][new_v] = (tp[0], tp[1] + fn + 1)
		assert(len(res[city]['v']) >= len(res[city]['vd']))

	e1 = [[r_idx2a_idx[s], r_idx2a_idx[t], score] for s, t, score in dt['edges']]
	e2 = [[r_idx2a_idx[t], r_idx2a_idx[s], score] for s, t, score in dt['edges']]
	res[city]['e'].extend(e1)
	res[city]['e'].extend(e2)


from PIL import Image, ImageDraw
os.popen('mkdir out_graph_big')
time.sleep(1)
for k in res:
	print(k)
	v = res[k]['v']
	e = res[k]['e']
	if False:
		img = Image.new('P', (4096, 4096), color = 0)
		draw = ImageDraw.Draw(img)
		for s, t, score in e:
			if score > 0.75:
				draw.line([item * (sel_size / 28) + 4096 for item in v[s] + v[t]], width = 1, fill = 255)
		img.show()
		input()
		continue
	for th in np.arange(0.5, 1.0, 0.05):
		with open('./out_graph_big/%s_%.2lf.out.graph' % (k, th), 'w') as f:
			for item in v:
				x, y = round(item[0] * (sel_size / 28)), round(item[1] * (sel_size / 28))
				f.write('%d %d\n' % (x, y))
			f.write('\n')
			for s, t, score in e:
				if s != t and score >= th:
					f.write('%d %d\n' % (s, t))

