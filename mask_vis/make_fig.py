from PIL import Image, ImageDraw

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
	return v, e

cities = [
	'boston', 'chicago', 'amsterdam', 'denver', 'kansas city',
	'la', 'montreal', 'new york', 'paris', 'pittsburgh',
	'saltlakecity', 'san diego', 'tokyo', 'toronto', 'vancouver'
]
biases = [
	[ 1, -1, 3, 1], [-1, -2, 1, 0], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1],
	[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1],
	[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]
]
dt_files = [
	'../out_ours_cu_remove/%s_0.70.out.graph',
	'../out_roadtracer/%s.0.3_0.3.newreconnect.graph',
	'../out_deep/%s.fix.connected.graph'
]
gt_file = '../dataset/data/graphs/%s.graph'
img_path = '../dataset/data/imagery/%s_%d_%d_sat.png'

for city, bias in zip(cities, biases):
	gtv, gte = read_graph(gt_file % city)
	gt_img = {}
	for x in range(bias[0], bias[2]):
		for y in range(bias[1], bias[3]):
			gtp = [(px - x * 4096, py - y * 4096) for px, py in gtv]
			img = Image.open(img_path % (city, x, y))
			draw = ImageDraw.Draw(img)
			for u, v in gte:
				draw.line(gtp[u] + gtp[v], fill = TABLEAU20[0], width = 4)
			gt_img[(x, y)] = img
	for dt_file in dt_files:
		who = dt_file.split('/')[1].replace('out_', '')
		print(city, who)
		dtv, dte = read_graph(dt_file % city)
		for x in range(bias[0], bias[2]):
			for y in range(bias[1], bias[3]):
				dtp = [(px - x * 4096, py - y * 4096) for px, py in dtv]
				temp_img = gt_img[(x, y)].copy()
				draw = ImageDraw.Draw(temp_img)
				for u, v in dte:
					draw.line(dtp[u] + dtp[v], fill = TABLEAU20_DEEP[1], width = 16)
				temp_img.save('%s_%d_%d_%s.png' % (city, x, y, who))



