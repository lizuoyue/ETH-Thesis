import random
import numpy as np
import PIL
from PIL import Image, ImageDraw
import downloadBuilding
import matplotlib.pyplot as plt
BATCH_SIZE = 3
def show(img):
	plt.imshow(img)
	plt.show()
if __name__ == '__main__':
	obj = downloadBuilding.BuildingListConstructor(range_vertices = (4, 4), filename = './buildingList.npy')
	print(obj.building)
	quit()
	data = obj.getImage(BATCH_SIZE)
	img = [np.array(item[0])[...,0:3]/255.0 for item in data] # batch_size 224 224 3
	single = [item[5] for item in data] # batch_size num_vertices+1 28 28
	single_true = np.transpose(np.array(single), axes = [0, 2, 3, 1]) # batch_size 28 28 num_vertices+1
	end_true = np.array([[0,0,0,0,1] for i in range(BATCH_SIZE)]) # batch_size num_vertices+1
	boundary_true = [item[3] for item in data]
	vertices_true = [item[4] for item in data]

	for i in range(BATCH_SIZE):
		show(img[i])
		show(boundary_true[i])
		show(vertices_true[i])
		for j in range(len(single[i])):
			show(single[i][j])

	
