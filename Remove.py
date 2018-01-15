import os, sys
import numpy as np

if __name__ == '__main__':
	if False:
		building_id_list = os.popen('ls ../Chicago').read().strip().split('\n')
		for i, bid in enumerate(building_id_list):
			print(i, bid)
			try:
				with open('../Chicago/%s/shift.txt' % bid) as f:
					lines = f.readlines()
					edge_prob, _ = lines[1].strip().split()
					edge_prob = float(edge_prob)
				if edge_prob < float(sys.argv[1]):
					os.system('rm -r ../Chicago/%s' % bid)
			except:
				os.system('rm -r ../Chicago/%s' % bid)
	if True:
		area_id_list = os.popen('ls ../Chicago_Area').read().strip().split('\n')
		count_pos = 0
		count_neg = 0
		for i, aid in enumerate(area_id_list):
			with open('../Chicago_Area/%s/polygons.txt' % aid) as f:
				lines = f.readlines()
				polygons = []
				for line in lines:
					if line.strip() != '':
						if line.strip() == '%':
							polygons.append([])
						else:
							x, y = line.strip().split()
							polygons[-1].append((int(x), int(y)))

				gt_boxes = []
				for polygon in polygons:
					w, h = 640, 640
					p = np.array(polygon, np.int32)
					l = max(0, p[:, 0].min())
					u = max(0, p[:, 1].min())
					r = min(w, p[:, 0].max())
					d = min(h, p[:, 1].max())
					if r > l and d > u:
						gt_boxes.append([u, l, d, r])
				count_neg += len(gt_boxes) == 0
				count_pos += len(gt_boxes) > 0
			print(i, aid, count_pos, count_neg)
			# os.system('rm -r ../Chicago_Area/%s' % aid)

