import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from skimage import io

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

def read_graph(filename):
	with open(filename) as f:
		lines = f.readlines()
	for i, line in enumerate(lines):
		if line == '\n':
			sep = i
			break
	v = lines[:sep]
	e = lines[sep+1:]
	v = [tuple(float(item) for item in line.strip().split()) for line in v]
	e = [tuple(int(item) for item in line.strip().split()) for line in e]
	nb = [set() for _ in v]
	for s, t in e:
		nb[s].add(t)
		nb[t].add(s)
	return v, e, nb

def judge(xlim, ylim, vs, vt):
	xmin, xmax = xlim
	ymax, ymin = ylim
	vsx, vsy = vs
	vtx, vty = vt
	vxmin, vxmax = min(vsx, vtx), max(vsx, vtx)
	vymin, vymax = min(vsy, vty), max(vsy, vty)

	flag1 = not vxmax < xmin
	flag2 = not xmax < vxmin
	flag3 = not vymax < ymin
	flag4 = not ymax < vymin

	return flag1 and flag2 and flag3 and flag4

def judge_v(xlim, ylim, v):
	xmin, xmax = xlim
	ymax, ymin = ylim
	vx, vy = v
	flag1 = (xmin <= vx and vx <= xmax)
	flag2 = (ymin <= vy and vy <= ymax)
	return flag1 and flag2

if __name__ == '__main__':
	cities = ['amsterdam', 'pittsburgh', 'la']
	biases = [(-1, 0), (0, -1), (-1, -1)]
	crop_x = [(1900, 3400), (3150, 3850), ( 256,  956)]
	crop_y = [(1250,  550), (4096, 2596), (4000, 2500)]
	# crop_x = [(1900, 2600), (3150, 3850), ( 256,  956)]
	# crop_y = [(1250,  550), (4096, 3396), (4000, 3300)]
	files = [
		# ('gt', '../dataset/data/graphs/%s.graph'),
		('ours', '../out_ours_cu_remove_clean/%s_0.70.out.graph'),
		('rt', '../out_roadtracer/%s.0.3_0.3.newreconnect.graph'),
		('drm', '../out_deep/%s.fix.connected.graph')
	]
	for city, (bx, by), cx, cy in zip(cities, biases, crop_x, crop_y):
		for name, file in files:
			print(city, name)
			vgt, egt, nbgt = read_graph('../dataset/data/graphs/%s.graph' % city)
			vgt = [(x - bx * 4096, y - by * 4096) for x, y in vgt]
			vdt, edt, nbdt = read_graph(file % city)
			vdt = [(x - bx * 4096, y - by * 4096) for x, y in vdt]
			img = io.imread('../dataset/data/imagery/%s_%d_%d_sat.png' % (city, bx, by))
			# img = np.ones((4096, 4096, 3), np.uint8) * 255
			fig, ax = plt.subplots(1, figsize = (32, 32), dpi = 128)
			ax.imshow(img)
			ax.axis('off')
			c1 = tuple(np.array(TABLEAU20[0]) / 255.0)
			c2 = tuple(np.array(TABLEAU20_DEEP[1]) / 255.0)
			c3 = tuple(np.array(TABLEAU20_DEEP[0]) / 255.0)
			for s, t in egt:
				if judge(cx, cy, vgt[s], vgt[t]):
					p = Polygon(np.array([vgt[s], vgt[t]]), facecolor = 'None', edgecolor = c1 + (1, ), linewidth = 8, joinstyle = 'round')
					ax.add_patch(p)
			for s, t in edt:
				if name == 'ours' and (len(nbdt[s]) <= 1 or len(nbdt[t]) <= 1):
					continue
				if judge(cx, cy, vdt[s], vdt[t]):
					p = Polygon(np.array([vdt[s], vdt[t]]), facecolor = 'None', edgecolor = c2 + (1, ), linewidth = 16, joinstyle = 'round')
					ax.add_patch(p)
			# for idx, v in enumerate(vdt):
			# 	if name == 'ours' and len(nbdt[idx]) <= 1:
			# 		continue
			# 	if judge_v(cx, cy, v):
			# 		p = Ellipse(v, width = 20, height = 20, facecolor = c3 + (1, ))
			# 		ax.add_patch(p)
			plt.xlim(cx)
			plt.ylim(cy)
			plt.savefig('%s_%s.pdf' % (city, name))



