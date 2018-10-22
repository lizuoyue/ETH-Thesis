import math
import numpy as np
from PIL import Image, ImageDraw
import io, os, sys, time
import requests, traceback
import Config
from UtilityGeography import *

config = Config.Config()
with open('GoogleMapsAPIsKeys.txt', 'r') as f:
	keys = [item.strip() for item in f.readlines()]

def DownloadMap(city_name, city_info):
	city_poly = city_info['map_area']
	poly = np.array(city_poly)
	minlat, maxlat = poly[:, 0].min(), poly[:, 0].max()
	minlon, maxlon = poly[:, 1].min(), poly[:, 1].max()
	z, s = city_info['map_zoom'], city_info['map_scale']
	wg, hg = city_info['map_size']
	w, h = s * wg, s * hg

	bbox = BoundingBox(minlon, maxlat, w, h, z, s) # Upper left corner
	s_lon, s_lat = bbox.relativePixelToLonLat(w, h)
	tl, lon_list = s_lon, [] # Lon list from left to right
	while tl < maxlon:
		lon_list.append(tl)
		tl += (s_lon - bbox.tc_lon)

	tl, lat_list = s_lat, [] # Lat list from up to down
	while tl > minlat:
		lat_list.append(tl)
		bbox = BoundingBox(s_lon, tl, w, h, z, s)
		_, tl = bbox.relativePixelToLonLat(w, h)
		if len(lat_list) > 1 and tl - lat_list[-1] > lat_list[-1] - lat_list[-2]:
			_, tl = bbox.relativePixelToLonLat(w, h + 1)

	assert(lon_list[0] > minlon)
	assert(lon_list[-1] < maxlon)
	assert(lat_list[0] < maxlat)
	assert(lat_list[-1] > minlat)

	w_img, h_img = len(lon_list) * 10 + 10, len(lat_list) * 10 + 10
	poly_img = (poly - np.array([minlat, minlon])) / np.array([maxlat - minlat, maxlon - minlon]) * np.array([h_img, w_img])
	polygon = [(poly_img[i, 1], h_img - poly_img[i, 0]) for i in range(poly_img.shape[0])]
	img = Image.new('P', (w_img, h_img), color = 0)
	draw = ImageDraw.Draw(img)
	draw.polygon(polygon, fill = 255, outline = 0)
	img_valid = np.array(img)
	draw.polygon(polygon, fill = 0, outline = 255)

	coo_list = []
	train_val_test = [0, 0, 0]
	for lon in lon_list:
		for lat in lat_list:
			x = math.floor((lon - minlon) / (maxlon - minlon) * w_img)
			y = h_img - math.floor((lat - minlat) / (maxlat - minlat) * h_img)
			if img_valid[y, x] > 128:
				coo_list.append((lat, lon))
				train_val_test[city_info['val_test'](lon, lat)] += 1
				draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill = 255, outline = 255)
		img.save('%sTemp.png' % city_name)
	print('Train/Val/Test Num:', train_val_test)

	if not os.path.exists(city_name):
		os.popen('mkdir %sMap' % city_name)

	pad = config.PAD * s
	d = {}
	for seq, (c_lat, c_lon) in enumerate(coo_list):
		while True:
			try:
				request_str = \
					'https://maps.googleapis.com/maps/api/staticmap?' 						+ \
					'maptype=%s&' 			% 'satellite' 									+ \
					'center=%.9lf,%.9lf&' 	% (c_lat, c_lon) 								+ \
					'zoom=%d&' 				% z 											+ \
					'size=%dx%d&' 			% (wg + config.PAD * 2, hg + config.PAD * 2)	+ \
					'scale=%d&' 			% s 											+ \
					'format=%s&' 			% 'png32' 										+ \
					'key=%s' 				% keys[seq % len(keys)]
				img_data = requests.get(request_str).content
				break
			except:
				e1, e2, e3 = sys.exc_info()
				traceback.print_exception(e1, e2, e3)
				time.sleep(3)
		d[seq] = {'center': (c_lat, c_lon), 'zoom': z, 'size': (wg + config.PAD * 2, hg + config.PAD * 2), 'scale': s}
		img = Image.open(io.BytesIO(img_data))
		img = np.array(img)[pad: h + pad, pad: w + pad, ...]
		Image.fromarray(img).save('%sMap/%s.png' % (city_name, str(seq).zfill(6)))
		print(seq, len(coo_list))
		np.save('%sMapInfo.npy' % city_name, d)
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	DownloadMap(city_name, config.CITY_INFO[city_name])
