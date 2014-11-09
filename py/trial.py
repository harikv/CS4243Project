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
import csv
from matplotlib.path import Path
from projection import return_projected_point, degtorad

# model = []
# polygon = []

model = [((0,0), np.array([-10, -27, 40]), (149,165,199)), ((0,0), np.array([15, -27, 40]), (65,105,164)), \
         ((0,0), np.array([-10, 30, -3]), (0,0,0)), ((0,0), np.array([15, 30, -3]), (0,0,0)), \
         ((0,0), np.array([-10, 30, 40]), (0,0,0)), ((0,0), np.array([15, 30, 40]), (0,0,0))]

polygon = [(320.0, 502.0), (657.0, 502.0), (931.0, 528.0), (0.0, 528.0)]
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
    with open('points.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for index in range(len(model)):
            writer.writerow([model[index][0][0], model[index][0][0], model[index][1][0], model[index][1][1], model[index][1][2], model[index][2][0], model[index][2][1], model[index][2][2]])

    # print model
    # print [m[1] for m in model]
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
        projected_points.append(projected_point)
    pixel_coords = quantize(projected_points)
    print len(model)
    for (pt, color) in zip(pixel_coords, [m[2] for m in model]):
        # print pt, color
        out_img[pt[0]][pt[1]] = np.array(list(color))
    cv2.imwrite("../test.jpg", out_img)    

def quantize(input_list):
    max_x = max(input_list[0])
    max_y = max(input_list[1])
    min_x = min(input_list[0])
    min_y = min(input_list[1])
    x_ratio = (max_x - min_x)
    y_ratio = (max_y - min_y)
    print x_ratio
    print y_ratio
    global out_img 
    out_img = np.zeros(shape=(x_ratio+1, y_ratio+1, 3))
    output_list = []
    for (x,y) in input_list:
        print x, y
        output_list.append([int(((x - min_x)*x_ratio*10000)/(max_x - min_x))/10000, int(((y - min_y)*y_ratio*10000)/(max_y - min_y))/10000])
    return output_list

im = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
img = cv2.imread("../project_rs.jpg",cv2.CV_LOAD_IMAGE_COLOR)
img_out = cv2.imread("../white.jpg",cv2.CV_LOAD_IMAGE_COLOR)
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