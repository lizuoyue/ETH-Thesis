import numpy as np
import PIL
from PIL import Image, ImageDraw

def plotPolygon(img_size = (224, 224), num_vertices = 6):
	# Set image parameters
	num_row = img_size[0]
	num_col = img_size[1]
	half_x = int(num_col / 2)
	half_y = int(num_row / 2)
	img_size_s = (int(num_row / 8), int(num_col / 8))

	# Set polygon parameters
	center_r = int(min(num_row, num_col) / 30)
	polygon_size = int(min(num_row, num_col) * 0.35)
	delta_angle = np.pi * 2 / num_vertices
	angle = np.random.uniform(0.0, delta_angle)

	# Determin the center of polygon
	center_x = half_x + np.random.randint(-center_r, center_r)
	center_y = half_y + np.random.randint(-center_r, center_r)

	# Determin the polygon vertices
	polygon = []
	polygon_s = []
	for i in range(num_vertices):
		r = polygon_size * np.random.uniform(0.7, 1.1)
		px = int(center_x + r * np.sin(angle))
		py = int(center_y + r * np.cos(angle))
		polygon.append((px, py))
		polygon_s.append((int(px / 8), int(py / 8)))
		angle += delta_angle * np.random.uniform(0.9, 1.1)

	# Draw polygon
	color = (255, 0, 0)
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	draw.polygon(polygon, fill = color, outline = color)
	boundary = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	vertices = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)

	# Add noise
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = Image.fromarray(img)
	
	# Show
	if False:
		img.show()
		boundary.show()
		vertices.show()

	# Return
	img = np.array(img) / 255.0
	boundary = np.array(boundary) / 255.0
	vertices = np.array(vertices) / 255.0
	return polygon, img, boundary, vertices

if __name__ == '__main__':
	for i in range(2):
		plotPolygon(num_vertices = 6)



