import os, time
import numpy as np

cities = [
	'boston', 'chicago', 'amsterdam', 'denver',
	'la', 'montreal', 'paris', 'pittsburgh',
	'saltlakecity', 'tokyo', 'toronto', 'vancouver',
	'new\\ york', 'kansas\\ city', 'san\\ diego'
]

def extract(s):
	lines = s.split('\n')
	num_jun_gt = int(lines[0].split()[0])
	info = lines[2].split(',')
	num_total = int(info[0].split()[0])
	num_yes = int(info[1].split()[0])
	num_no = int(info[3].split()[0])
	return num_jun_gt, num_yes, num_no

errs, recs = [], []
for th in np.arange(0.5, 1.0, 0.05):
	total = [0, 0, 0]
	for city in cities:
		filename = 'go run junction_metric.go dataset/data/graphs/%s.graph out_ours_uc/%s_%.2lf.out.graph %s' % \
		(city, city, th, city.replace('\\ ', ''))
		num_jun_gt, num_yes, num_no = extract(os.popen(filename).read())
		total[0] += num_jun_gt
		total[1] += num_yes
		total[2] += num_no
	pre = total[1] / (total[1] + total[2])
	rec = total[1] / total[0]
	errs.append(1 - pre)
	recs.append(rec)
	print(th, rec, 1 - pre)
