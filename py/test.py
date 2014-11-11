import cv2
import numpy as np
from matplotlib.path import Path
from projection import return_projected_point, degtorad
import math
import os
import csv


def getFileName(poly, model3d, polygonCount):
	string = ''
	for x,y in zip(poly, model3d):
		string += str(x)+str(y)
	string += str(polygonCount)
	return str(hash(string))

def writeModelToFile(fileName):
	with open(fileName+'.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow([x for x in polygon])
		writer.writerow([x for x in corr_3d])
		writer.writerow([polygonCount])
		for key, value in model.iteritems():
			writer.writerow([key[0], key[1], key[2], value[0], value[1], value[2]])

def fileForPolygon(fileName):
	return bool(os.path.exists(''+fileName+'.csv'))

def loadIntoModel(fileName):
	with open(fileName+'.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		lineNumber = 0
		for row in reader:
			if (lineNumber > 2):
				model[(float(row[0]), float(row[1]), float(row[2]))] = (float(row[3]), float(row[4]), float(row[5]))
			else:
				lineNumber += 1

def compare_floats(f1, f2):
	return abs(f1 - f2) <= 0.00001

def compare_color(color1, color2):
	if(compare_floats(color1[0],color2[0]) and compare_floats(color1[1],color2[1]) and compare_floats(color1[2],color2[2])):
		return True
	return False

def createNullImage(shape):
	newImage = []
	null_color = np.array([256.00, 256.00, 256.00])
	for row in range(shape[0]):
		newImage.append([])
		for column in range(shape[1]):
			newImage[row].append(null_color)
	return np.asarray(newImage)

def replace_null_img(img):
	num_rows = img.shape[0]
	num_cols = img.shape[1]
	null_color = np.array([256.00 ,256.00 ,256.00 ])
	for row in range(num_rows):
		for column in range(num_cols):
			if(compare_color(img[row][column], null_color)):
				img[row][column] = np.array([0.00 ,0.00 ,0.00 ])
	return img

def carryOutDithering(iteration, img):
	num_rows = img.shape[0]
	num_cols = img.shape[1]
	null_color = np.array([256.00,256.00,256.00])
	for index in range(iteration):
		print "Dithering Revision: ", (index+1)
		for row in range(num_rows):
			for column in range(num_cols):
				avg = np.array([0.00,0.00,0.00])
				count = 0
				countNull = 0
				if(compare_color(img[row][column],null_color)):
					if(row > 0 and column  > 0):
						# print img[row-1][column-1]
						count += 1
						if(not compare_color(img[row-1][column-1],null_color)):
							avg += img[row-1][column-1]
						else:
							countNull += 1
					if(row > 0):
						# print img[row-1][column]
						count += 1
						if(not compare_color(img[row-1][column],null_color)):
							avg += img[row-1][column]
						else:
							countNull += 1
					if(row > 0 and column < num_cols -1):
						# print img[row-1][column+1]
						count += 1
						if(not compare_color(img[row-1][column+1],null_color)):
							avg += img[row-1][column+1]
						else:
							countNull += 1
					if(column > 0):
						count += 1
						if(not compare_color(img[row][column-1],null_color)):
							avg += img[row][column-1]
						else:
							countNull += 1
					if(column > 0 and row < num_rows - 1):
						count += 1
						if(not compare_color(img[row+1][column-1],null_color)):
							avg += img[row+1][column-1]
						else:
							countNull += 1
					if(row < num_rows - 1):
						count += 1
						if(not compare_color(img[row+1][column],null_color)):
							avg += img[row+1][column]
						else:
							countNull += 1
					if(column < num_cols - 1 and row < num_rows - 1):
						count += 1
						if(not compare_color(img[row+1][column+1],null_color)):
							avg += img[row+1][column+1]
						else:
							countNull += 1
					if(column < num_cols - 1):
						count += 1
						if(not compare_color(img[row][column+1],null_color)):
							avg += img[row][column+1]
						else:
							countNull += 1
					avg = avg/(count-(0.5*countNull))
					if(count != countNull):
						img[row][column] = avg
	return replace_null_img(img)

def getDivisions(pt1, pt2, granularity):
	vector = np.array([(x[1] - x[0]) for x in zip(pt1, pt2)])
	return [(np.array(pt1) + (vector * (i)/granularity)) for i in range(1, granularity + 1)]

def getExtraColors(p1, p2, p3, p4, c1, c2):
	div = polygonCount
	array3D1 = getDivisions(p2, p3, div)
	array3D2 = getDivisions(p1, p4, div)
	min_len = min(len(c1), len(c2))
	max_len = max(len(c1), len(c2))
	diff = max_len - min_len
	longer = c2
	shorter = c1
	extra = []
	if(len(c1) > len(c2)):
		longer = c1
		shorter = c2
	color_array = [getDivisions(np.asarray(color1, dtype='float64'), np.asarray(color2, dtype='float64'), div) for color1,color2 in zip(shorter[0:min_len+1], longer[int(diff/2):(int(diff/2)+min_len+1)])]
	color_array = np.transpose(np.array(color_array), [1,0,2])
	for i in range(div-1):
		numberHorizontalDivisions = int(min_len)
		array3DHorizontal = getDivisions(array3D1[i], array3D2[i], numberHorizontalDivisions)
		extra.extend(zip(array3DHorizontal, color_array[i]))
	return extra

def create_model(points):
	numberDivisions = abs(len(points))
	array3D1 = getDivisions(corr_3d[1], corr_3d[2], int(numberDivisions))
	array3D2 = getDivisions(corr_3d[0], corr_3d[3], int(numberDivisions))
	for index in range(len(points)):
		numberHorizontalDivisions = len(points[index])
		array3DHorizontal = getDivisions(array3D1[index], array3D2[index], numberHorizontalDivisions)
		colours = zip([img[x[1]][x[0]][0] for x in points[index]], [img[x[1]][x[0]][1] for x in points[index]], [img[x[1]][x[0]][2] for x in points[index]])
		extraTuples = []
		if(index < len(points) - 1):
			extraColors = zip([img[x[1]][x[0]][0] for x in points[index+1]], [img[x[1]][x[0]][1] for x in points[index+1]], [img[x[1]][x[0]][2] for x in points[index+1]])
			extraTuples = getExtraColors(array3D1[index], array3D2[index], array3D1[index+1], array3D2[index+1], colours, extraColors)
		for point, colours in zip(array3DHorizontal, colours):
			model[tuple(point)] = colours
		for point, colours in extraTuples:
			model[tuple(point)] = colours

def contains(element, list):
	try:
		return bool(list.index(element))
	except ValueError:
		return False

def processPolygon(polygon):
	length = len(polygon)
	polygon.append((0.0, 0.0))
	codes = [Path.MOVETO]
	for index in range(length-1):
		codes.append(Path.LINETO)
	codes.append(Path.CLOSEPOLY)
	path = Path(polygon, codes)
	points = []
	if mode == 'V':
		for index in range(rows):
			row = [(x, index) for x in range(columns)]
			check = path.contains_points(row)
			temp_points = ([row[i] for i, j in enumerate(check) if j == True and not contains(row[i], polygon)])
			if(len(temp_points) > 0):
				points.append(temp_points)
	else:
		for index in range(columns):
			col = [(index, x) for x in range(rows)]
			check = path.contains_points(col)
			temp_points = [col[i] for i, j in enumerate(check) if j == True and not contains(col[i], polygon)]
			if(len(temp_points) > 0):
				points.append(temp_points)
	create_model(points)

def projectModel():
	projected_points = {}
	# camera_orientation = np.matrix([[1/math.sqrt(2),-1/math.sqrt(2),0],[0,0,-1],[1/math.sqrt(2),1/math.sqrt(2),0]])
	camera_orientation = np.matrix([[1.00,0.00,0.00],[0.00,0.00,1.00],[0.00,1.00,0.00]])
	for pt in model:
		projected_point = return_projected_point(np.array(pt), np.array([0.00, -1.10 , -0.50]), viewing_angle_in_radians, camera_orientation)
		if projected_point is not None:
			projected_points[pt] = projected_point
	pixels = quantize(projected_points)
	return pixels

def createImage(pixels, model):
	max_x = max([x[0] for key, x in pixels.iteritems()])
	max_y = max([x[1] for key, x in pixels.iteritems()])
	for key, (x, y) in pixels.iteritems():
		if(y>(viewport[0]-1) or x>(viewport[1]-1)):
			continue
		out_img[-1*(y)][x] = np.array(model[key])

def quantize(input_list):
	min_x = min([x[0] for key, x in input_list.iteritems()])
	min_y = min([x[1] for key, x in input_list.iteritems()])
	# max_x = max([x[0] for key, x in input_list.iteritems()])
	# max_y = max([x[1] for key, x in input_list.iteritems()])
	output_list = {}
	for key, (x,y) in input_list.iteritems():
		output_list[key] = (int(((x - min_x)*10000*(70)))/10000, int(((y - min_y)*10000*(5)))/10000)
	return output_list

img = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
viewport = (700, 933)
# viewport = (300, 300)
out_img = createNullImage(viewport)
rows = img.shape[0]
columns = img.shape[1]
mode = 'V'
model = {}
viewing_angle_in_radians = degtorad(90)

polygon = [(707, 387),
		   (707, 526),
		   (),
		   ()]
polygon = [(320,502),
		   (657,502),
		   (933,700),
		   (0,700)]
corr_3d = [(-1.00, 0.00, -3.00),
		   (1.00,0.00,-3.00),
		   (1.00,-1.00,-3.00),
		   (-1.00,-1.00,-3.00)]
polygonCount = 1
fileName = 'lawn'+getFileName(polygon, corr_3d, polygonCount)
if(fileForPolygon(fileName)):
	loadIntoModel(fileName)
else:
	processPolygon(polygon)
	writeModelToFile(fileName)
# polygon = [(150, 105), (150, 230), (245, 185), (245, 60)]
# corr_3d = [(0,0,0),(0,0,-1),(1,0,-1),(1,0,0)]
# processPolygon(polygon)

# projecting and carrying out quantization
quantized_pixels = projectModel()

# coloring plane image
createImage(quantized_pixels, model)

# Carrying out dilation
out_img = carryOutDithering(2, out_img)

# writing image to file
cv2.imwrite("test.jpg", out_img)
