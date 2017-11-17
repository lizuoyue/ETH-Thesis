import numpy as np
import sys, math, random
from PIL import Image, ImageDraw
# PIL.ImageDraw: 0-based, (col_idx, row_idx) if taking image as matrix

def pil2np(image, show):
	if show:
		import matplotlib.pyplot as plt
	img = np.array(image, dtype = np.float32) / 255.0
	if show:
		plt.imshow(img)
		plt.show()
	return img

def plotPolygon(img_size = (224, 224), num_vertices = 6, show = False):

	# Set image parameters
	num_row = img_size[0]
	num_col = img_size[1]
	half_x = math.floor(num_col / 2)
	half_y = math.floor(num_row / 2)
	img_size_s = (math.floor(num_row / 8), math.floor(num_col / 8))

	# Set polygon parameters
	epsilon = 1.0 / num_vertices
	center_r = math.floor(min(num_row, num_col) * 0.05) # <- Decide polygon's center
	polygon_size = math.floor(min(num_row, num_col) * 0.35) # <- Decide polygon's size
	delta_angle = np.pi * 2 * epsilon
	angle = np.random.uniform(0.0, delta_angle) # <- Decide polygon's first vertex

	# Determin the center of polygon
	center_x = half_x + np.random.randint(-center_r, center_r)
	center_y = half_y + np.random.randint(-center_r, center_r)

	# Determin the polygon vertices
	polygon = []
	polygon_s = []
	for i in range(num_vertices):
		r = polygon_size * np.random.uniform(0.8, 1.1) # <- Decide polygon's size range
		px = math.floor(center_x + r * np.cos(angle))
		py = math.floor(center_y - r * np.sin(angle)) # <- Decide polygon's order (counterclockwise)
		polygon.append((px, py))
		polygon_s.append((math.floor(px / 8), math.floor(py / 8)))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon) # <- Decide polygon's vertices
	first_idx = random.choice([i for i in range(num_vertices)])
	polygon = polygon[first_idx:] + polygon[:first_idx]
	polygon_s = polygon_s[first_idx:] + polygon_s[:first_idx]

	# Draw polygon
	color = (255, 0, 0)
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	draw.polygon(polygon, fill = color, outline = color)

	# Add noise to the orginal image
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = Image.fromarray(img)
	img = pil2np(img, show)

	# Draw boundary
	boundary = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	boundary = pil2np(boundary, show)

	# Draw vertices
	vertices = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)
	vertices = pil2np(vertices, show)

	# Draw each vertex
	vertex_list = []
	for i in range(num_vertices):
		vertex = Image.new('P', img_size_s, color = 0)
		draw = ImageDraw.Draw(vertex)
		draw.point([polygon_s[i]], fill = 255)
		vertex = pil2np(vertex, show)
		vertex_list.append(vertex)
	vertex_list.append(np.zeros(img_size_s, dtype = np.float32))
	vertex_list = np.array(vertex_list)

	# Return
	if show:
		print(img.shape)
		print(boundary.shape)
		print(vertices.shape)
		print(vertex_list.shape)
	return img, boundary, vertices, vertex_list

if __name__ == '__main__':
	for i in range(1):
		plotPolygon(num_vertices = 7, show = False)

