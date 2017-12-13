import os, zipfile

def removeBuilding(archive, building_id, th = 0.9):
	path = 'Chicago/' + str(building_id)

	#
	lines = archive.read(path + '/shift.txt').decode('utf-8').split('\n')
	edge_prob, _ = lines[1].strip().split()
	edge_prob = float(edge_prob)

	#
	if edge_prob < th:
		os.system('zip -d ../Chicago.zip \"*/%s/*\"' % building_id)
	return

if __name__ == '__main__':
	# archive = zipfile.ZipFile('../Chicago.zip', 'r')
	# building_id_set = set()
	# for filename in archive.namelist():
	# 	if filename.startswith('__MACOSX'):
	# 		continue
	# 	parts = filename.split('/')
	# 	if len(parts) == 3:
	# 		building_id_set.add(int(parts[1]))
	# for bid in building_id_set:
	# 	print(bid)
	# 	removeBuilding(archive, bid)

	building_id_list = os.popen('ls ../Chicago').read().strip().split('\n')
	for i, bid in ennumerate(building_id_list):
		print(i, bid)
		with open('../Chicago/%s/shift.txt' % bid) as f:
			lines = f.readlines()
			edge_prob, _ = lines[1].strip().split()
			edge_prob = float(edge_prob)
		if edge_prob < 0.9:
			os.system('rm -r ../Chicago/%s' % bid)
