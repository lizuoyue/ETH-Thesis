import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def generatePolygon(img_size = (224, 224), padding_div = 16, polygon_type = 'tri'):
	n_row = img_size[0]
	n_col = img_size[1]
	half_x = int(n_col / 2)
	half_y = int(n_row / 2)
	pad_x = int(n_col / padding_div)
	pad_y = int(n_row / padding_div)
	if polygon_type == 'tri':
		if np.random.rand() > 0.5:
			p1x = np.random.randint(pad_x, half_x - pad_x)
			p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
			p3x = np.random.randint(pad_x, n_col - pad_x)
			p1y = np.random.randint(pad_y, half_y - pad_y)
			p2y = np.random.randint(pad_y, half_y - pad_y)
			p3y = np.random.randint(half_y + pad_y, n_row - pad_y)
		else:
			p1x = np.random.randint(pad_x, half_x - pad_x)
			p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
			p3x = np.random.randint(pad_x, n_col - pad_x)
			p1y = np.random.randint(half_y + pad_y, n_row - pad_y)
			p2y = np.random.randint(half_y + pad_y, n_row - pad_y)
			p3y = np.random.randint(pad_y, half_y - pad_y)
		polygon = [(p1x, p1y), (p2x, p2y), (p3x, p3y)]
	elif polygon_type == 'qua':
		p1x = np.random.randint(pad_x, half_x - pad_x)
		p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
		p3x = np.random.randint(half_x + pad_x, n_col - pad_x)
		p4x = np.random.randint(pad_x, half_x - pad_x)
		p1y = np.random.randint(pad_y, half_y - pad_y)
		p2y = np.random.randint(pad_y, half_y - pad_y)
		p3y = np.random.randint(half_y + pad_y, n_row - pad_y)
		p4y = np.random.randint(half_y + pad_y, n_row - pad_y)
		polygon = [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)]
	else:
		return
	img = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(img)
	draw.polygon(polygon, fill = (255, 0, 0), outline = (255, 0, 0))
	noise = np.random.normal(0, 50, (n_row, n_col, 3))
	background = np.array(img)
	new_img = background + noise
	new_img = np.array((new_img - np.amin(new_img)) / (np.amax(new_img) - np.amin(new_img)) * 255.0, dtype = np.uint8)
	new_img = Image.fromarray(new_img)
	new_img.show()

if __name__ == '__main__':
	for i in range(20):
		generatePolygon(polygon_type = 'tri')