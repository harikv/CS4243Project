import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from projection import return_projected_point, degtorad
import math

model = []

# polygon = [(320.0, 502.0), (657.0, 502.0), (931.0, 528.0), (0.0, 528.0)]
polygon = [(55, 70)]
drawing = False # true if mouse is pressed
mode = 'V' # if True, draw rectangle. Press 'm' to toggle to curve
parallelX = False
parallelY = False
viewing_angle_in_radians = degtorad(90)
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,parallelX, parallelY
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if startPolygon and len(polygon)!=4 :
            ix = x
            iy = y
            if parallelX==True:
                ix = int(polygon[-1][0])
                parallelX = False
            if parallelY==True:
                iy = int(polygon[-1][1])
                parallelY = False
            cv2.circle(img,(ix,iy),1,(0,0,255),-1)
            print (ix, iy)
            polygon.append((ix*1.0,iy*1.0))

def getDivisions(pt1, pt2, granularity):
    vector = np.array([(x[1] - x[0]) for x in zip(pt1, pt2)])
    return [(np.array(pt1) + (vector * (float(i)/granularity))) for i in range(1, granularity + 1)]

def startScanLine(points):
    numberDivision1 = 0
    numberDivision2 = 0
    if mode == 'V':
        numberDivision1 = polygon[2][1] - polygon[1][1]
        numberDivision2 = polygon[3][1] - polygon[0][1]
    elif mode == 'H':
        numberDivision1 = polygon[2][0] - polygon[1][0]
        numberDivision2 = polygon[3][0] - polygon[0][0]
    if numberDivision1 != numberDivision2:
        print "error"
        return
    in_polygon = [(-10, 0, -3), (10, 0, -3), (10, -10, -3), (-10, -10, -3)]
    # x_coords = []
    # y_coords = []
    # z_coords = []
    # for i in range(len(polygon)-1):
    #     x = int(raw_input('Enter the '+str(i+1)+' x coordinate input\n'))
    #     y = int(raw_input('Enter the '+str(i+1)+' y coordinate input\n'))
    #     z = int(raw_input('Enter the '+str(i+1)+' z coordinate input\n'))
    #     x_coords.append(x)
    #     y_coords.append(y)
    #     z_coords.append(z)
    #     in_polygon.append((x,y,z))
    array3D1 = getDivisions(in_polygon[1], in_polygon[2], int(numberDivision1))
    array3D2 = getDivisions(in_polygon[0], in_polygon[3], int(numberDivision2))
    for index in range(len(points)):
        numberHorizontalDivisions = len(points[index])
        array3DHorizontal = getDivisions(array3D1[index], array3D2[index], numberHorizontalDivisions)
        model.extend(zip(points[index], array3DHorizontal, zip([img[x[1]][x[0]][0] for x in points[index]], [img[x[1]][x[0]][1] for x in points[index]], [img[x[1]][x[0]][2] for x in points[index]])))


def processPolygon():
    length = len(polygon)
    polygon.append((0., 0.))
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
            temp_points = ([row[i] for i, j in enumerate(check) if j == True])
            if(len(temp_points) > 0):
                points.append(temp_points)
    else:
        for index in range(columns):
            col = [(x, index) for x in range(rows)]
            check = path.contains_points(col)
            temp_points = [col[i] for i, j in enumerate(check) if j == True]
            if(len(temp_points) > 0):
                points.append(temp_points)
    startScanLine(points)

def projectModel():
    projected_points = []
    for pt in [m[1] for m in model]:
        projected_point = return_projected_point(pt, [0, -27.1, 0], viewing_angle_in_radians)
        if projected_point is not None:
            projected_points.append(projected_point)
    pixel_coords = quantize(projected_points)
    print len(model)
    for (pt, color) in zip(pixel_coords, [m[2] for m in model]):
        # print pt, color
        out_img[pt[1]][pt[0]] = np.array(list(color))
    cv2.imwrite("../test.jpg", out_img)    

def quantize(input_list):
    max_x = max([x[0] for x in input_list])
    max_y = max([x[1] for x in input_list])
    min_x = min([x[0] for x in input_list])
    min_y = min([x[1] for x in input_list])
    x_ratio = (min_x - max_x)
    y_ratio = (min_y - max_y)
    print x_ratio
    print y_ratio
    output_list = []
    for (x,y) in input_list:
        # print x, y
        output_list.append([int(((x - max_x)*10000*932)/x_ratio)/10000, int(((y - max_y)*10000*699)/y_ratio)/10000])
    return output_list

im = cv2.imread("../cube.jpg",cv2.CV_LOAD_IMAGE_COLOR)
img = cv2.imread("../cube.jpg",cv2.CV_LOAD_IMAGE_COLOR)
out_img = cv2.imread("../white.jpg",cv2.CV_LOAD_IMAGE_COLOR)
print img.shape
print img[0].shape
print img[1].shape
print img[2].shape
rows=img.shape[0]
columns=img.shape[1]
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
startPolygon = True
print 'Select Polygon'
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = 'V' if mode == 'H' else 'H'
        if(mode == 'V'):
            print "Mode change to Vertical"
        else:
            print "Mode change to Horizontal"
    elif k == ord('s'):
        print "Resetted"
        polygon = []
        startPolygon = True
        parallelX = False
        parallelY = False
        img = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
    elif k == ord('e'):
        print "Started Processing"
        processPolygon()
        projectModel()
    elif k == ord('o'):
        print 'Vertical Flag Toggled'
        parallelX = True
    elif k == ord('p'):
        print 'Horizontal Flag Toggled'
        parallelY = True
    elif k == 27:
        break

cv2.destroyAllWindows()