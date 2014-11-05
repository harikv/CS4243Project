# __author__ = 'Akaash'
#
# import cv2
# import numpy as np
#
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     print flags, param
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print x,y
#         cv2.circle(img,(x,y),2,(255,0,0), 1, lineType=8, shift = 0)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         print x,y
#
# # Create a black image, a window and bind the function to window
# img = cv2.imread('../project.jpeg', cv2.CV_LOAD_IMAGE_COLOR)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
#
#
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
model = []
polygon = []
drawing = False # true if mouse is pressed
mode = 'V' # if True, draw rectangle. Press 'm' to toggle to curve
parallelX = False
parallelY = False
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
    in_polygon = [(-10, 0, -3), (10, 0, -3), (10, -27, -3), (-10, -27, -3)]
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
    print model
    # perpendicularDirection = int(raw_input('Enter the perpendicular direction\n'))
    # diff = 1
    # start = 0
    # end = 0
    # perpendicularMax = 0
    # perpendicularMin = 0
    # if scanDirection == 0 :
    #     start = min(x_coords)
    #     end = max(x_coords)
    #     diff = max(x_coords) - min(x_coords)
    # elif scanDirection == 1:
    #     start = min(y_coords)
    #     end = max(y_coords)
    #     diff = max(y_coords) - min(y_coords)
    # elif scanDirection == 2:
    #     start = min(z_coords)
    #     end = max(z_coords)
    #     diff = max(z_coords) - min(z_coords)
    # if perpendicularDirection == 0 :
    #     perpendicularMax = max(x_coords)
    #     perpendicularMin =  min(x_coords)
    # elif perpendicularDirection == 1:
    #     perpendicularMax = max(y_coords)
    #     perpendicularMin =  min(y_coords)
    # elif perpendicularDirection == 2:
    #     perpendicularMax = max(z_coords)
    #     perpendicularMin =  min(z_coords)
    # pixelDifference = len(points)
    # Increase_factor = diff*1.0/pixelDifference
    # for i in np.arange(start, end, Increase_factor):
    #     print i

def processPolygon():
    print polygon
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


im = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
img = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
print img.shape
print img[0].shape
print img[1].shape
print img[2].shape
rows=img.shape[0]
columns=img.shape[1]
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
startPolygon = True
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
        polygon = []
        startPolygon = True
        img = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
    elif k == ord('e'):
        processPolygon()
    elif k == ord('o'):
        parallelX = True
    elif k == ord('p'):
        parallelY = True
    elif k == 27:
        break

cv2.destroyAllWindows()