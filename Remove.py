import os, sys, zipfile

if __name__ == '__main__':
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
