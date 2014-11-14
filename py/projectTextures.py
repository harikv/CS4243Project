import cv2
import numpy as np
from projection import return_projected_point, degtorad
from matplotlib.path import Path
import math
import os
import csv
from numpy import linalg as la
from draw_picture import get_corners_of_cut_texture, get_cutoff_points, add_dummy_point


def compare_floats(f1, f2):
    return abs(f1 - f2) <= 0.00001


def compare_color(color1, color2):
    if (compare_floats(color1[0], color2[0]) and compare_floats(color1[1], color2[1]) and compare_floats(color1[2],
                                                                                                         color2[2])):
        return True
    return False


def replace_null_img(img):
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    null_color = np.array([256.00, 256.00, 256.00])
    for row in range(num_rows):
        for column in range(num_cols):
            if (compare_color(img[row][column], null_color)):
                img[row][column] = np.array([0.00, 0.00, 0.00])
    return img


def carryOutDithering(iteration, img):
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    null_color = np.array([256.00, 256.00, 256.00])
    for index in range(iteration):
        print "Dithering Revision: ", (index + 1)
        for row in range(num_rows):
            for column in range(num_cols):
                avg = np.array([0.00, 0.00, 0.00])
                count = 0
                countNull = 0
                if (compare_color(img[row][column], null_color)):
                    if (row > 0 and column > 0):
                        # print img[row-1][column-1]
                        count += 1
                        if (not compare_color(img[row - 1][column - 1], null_color)):
                            avg += img[row - 1][column - 1]
                        else:
                            countNull += 1
                    if (row > 0):
                        # print img[row-1][column]
                        count += 1
                        if (not compare_color(img[row - 1][column], null_color)):
                            avg += img[row - 1][column]
                        else:
                            countNull += 1
                    if (row > 0 and column < num_cols - 1):
                        # print img[row-1][column+1]
                        count += 1
                        if (not compare_color(img[row - 1][column + 1], null_color)):
                            avg += img[row - 1][column + 1]
                        else:
                            countNull += 1
                    if (column > 0):
                        count += 1
                        if (not compare_color(img[row][column - 1], null_color)):
                            avg += img[row][column - 1]
                        else:
                            countNull += 1
                    if (column > 0 and row < num_rows - 1):
                        count += 1
                        if (not compare_color(img[row + 1][column - 1], null_color)):
                            avg += img[row + 1][column - 1]
                        else:
                            countNull += 1
                    if (row < num_rows - 1):
                        count += 1
                        if (not compare_color(img[row + 1][column], null_color)):
                            avg += img[row + 1][column]
                        else:
                            countNull += 1
                    if (column < num_cols - 1 and row < num_rows - 1):
                        count += 1
                        if (not compare_color(img[row + 1][column + 1], null_color)):
                            avg += img[row + 1][column + 1]
                        else:
                            countNull += 1
                    if (column < num_cols - 1):
                        count += 1
                        if (not compare_color(img[row][column + 1], null_color)):
                            avg += img[row][column + 1]
                        else:
                            countNull += 1
                    avg = avg / (count - (0.5 * countNull))
                    if (count != countNull):
                        img[row][column] = avg
    return replace_null_img(img)


def populate_texture_list(fileName):
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        info_row = next(reader)
        no_of_textures = int(info_row[0])
        for i in range(no_of_textures):
            texture_info = next(reader)
            height_texture = int(texture_info[2])
            width_texture = texture_info[1]
            texture_array = []
            for row in range(height_texture):
                texture_array.append([])
                next_row = next(reader)
                int_next_row = [eval(x) for x in next_row]
                texture_array[row] = int_next_row
            textures[texture_info[0]] = texture_array


def createNullImage(shape):
    newImage = []
    null_color = np.array([256.00, 256.00, 256.00])
    for row in range(shape[0]):
        newImage.append([])
        for column in range(shape[1]):
            newImage[row].append(null_color)
    return np.asarray(newImage)


def defineModel():
    # corr_3d = [(-20.00,20.00,-1.00),
    # (20.00,20.00,-1.00),
    # (20.00,20.00,19.00),
    # (-20.00,20.00,19.00)]
    # model.append({'set': corr_3d, 'pattern': 'sky1'})

    # corr_3d = [(-20.00,20.00,19.00),
    #        (20.00,20.00,19.00),
    #        (20.00,-20.00,19.00),
    #        (-20.00,-20.00,19.00)]
    # model.append({'set': corr_3d, 'pattern': 'sky_main'})

    # corr_3d = [(-20.00,-20.00,19.00),
    #        (20.00,-20.00,19.00),
    #        (20.00,-20.00,-1.00),
    #        (-20.00,-20.00,-1.00)]
    # model.append({'set': corr_3d, 'pattern': 'sky3'})

    # corr_3d = [(-20.00,20.00,-1.00),
    #        (-20.00,20.00,19.00),
    #        (20.00,-20.00,19.00),
    #        (20.00,-20.00,-1.00)]
    # model.append({'set': corr_3d, 'pattern': 'sky3'})

    # corr_3d = [(20.00,20.00,19.00),
    #        (20.00,20.00,-1.00),
    #        (20.00,-20.00,-1.00),
    #        (20.00,-20.00,19.00)]
    # model.append({'set': corr_3d, 'pattern': 'sky4'})

    # corr_3d = [(-20.00, 20.00, -1.00),
    #            (20.00,20.00,-1.00),
    #            (20.00,-10.00,-1.00),
    #            (-20.00,-10.00,-1.00)]
    # model.append({'set': corr_3d, 'pattern': 'ground'})

    corr_3d = [(-3.00, 0.00, 5.00),
               (3.00, 0.00, 5.00),
               (3.00, 0.00, -1.00),
               (-3.00, 0.00, -1.000)]
    model.append({'set': corr_3d, 'pattern': 'front'})

    corr_3d = [(-10.00, 0.00, -1.00),
               (10.00, 0.00, -1.00),
               (10.00, -10.00, -1.00),
               (-10.00, -10.00, -1.00)]
    model.append({'set': corr_3d, 'pattern': 'lawn'})

    corr_3d = [(-10.00, 0.00, 4.00),
               (-3.00, 0.00, 4.00),
               (-3.00, 0.00, -1.00),
               (-10.00, 0.00, -1.00)]
    model.append({'set': corr_3d, 'pattern': 'corridor'})

    corr_3d = [(3.00, 0.00, 4.00),
               (10.00, 0.00, 4.00),
               (10.00, 0.00, -1.00),
               (3.00, 0.00, -1.00)]
    model.append({'set': corr_3d, 'pattern': 'corridor'})

    corr_3d = [(10.00, 0.00, 5.00),
               (12.00, 0.00, 5.00),
               (12.00, 0.00, -1.00),
               (10.00, 0.00, -1.00)]
    model.append({'set': corr_3d, 'pattern': 'staircase'})
    for offset in [1, -1]:
        for index in range(2):
            corr_3d = [(10 * offset, -9.0, 9 - 5 * index),
                       (10 * offset, -10, 9 - 5 * index),
                       (10 * offset, -10, 4 - 5 * index),
                       (10 * offset, -9.0, 4 - 5 * index)]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(10 * offset, -8.0, 9 - 5 * index),
                       (10 * offset, -9.0, 9 - 5 * index),
                       (10 * offset, -9.0, 4 - 5 * index),
                       (10 * offset, -8.0, 4 - 5 * index)]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(10 * offset, -7.0, 9 - 5 * index),
                       (10 * offset, -8.0, 9 - 5 * index),
                       (10 * offset, -8.0, 4 - 5 * index),
                       (10 * offset, -7.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(11 * offset, -7.0, 9 - 5 * index),
                       (10 * offset, -7.0, 9 - 5 * index),
                       (10 * offset, -7.0, 4 - 5 * index),
                       (11 * offset, -7.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(12 * offset, -7.0, 9 - 5 * index),
                       (11 * offset, -7.0, 9 - 5 * index),
                       (11 * offset, -7.0, 4 - 5 * index),
                       (12 * offset, -7.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(12 * offset, -6.0, 9 - 5 * index),
                       (12 * offset, -7.0, 9 - 5 * index),
                       (12 * offset, -7.0, 4 - 5 * index),
                       (12 * offset, -6.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(12 * offset, -5.0, 9 - 5 * index),
                       (12 * offset, -6.0, 9 - 5 * index),
                       (12 * offset, -6.0, 4 - 5 * index),
                       (12 * offset, -5.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(12 * offset, -4.0, 9 - 5 * index),
                       (12 * offset, -5.0, 9 - 5 * index),
                       (12 * offset, -5.0, 4 - 5 * index),
                       (12 * offset, -4.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(11 * offset, -4.0, 9 - 5 * index),
                       (12 * offset, -4.0, 9 - 5 * index),
                       (12 * offset, -4.0, 4 - 5 * index),
                       (11 * offset, -4.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(10 * offset, -4.0, 9 - 5 * index),
                       (11 * offset, -4.0, 9 - 5 * index),
                       (11 * offset, -4.0, 4 - 5 * index),
                       (10 * offset, -4.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(10 * offset, -3.0, 9 - 5 * index),
                       (10 * offset, -4.0, 9 - 5 * index),
                       (10 * offset, -4.0, 4 - 5 * index),
                       (10 * offset, -3.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

            corr_3d = [(10 * offset, -2.0, 9 - 5 * index),
                       (10 * offset, -3.0, 9 - 5 * index),
                       (10 * offset, -3.0, 4 - 5 * index),
                       (10 * offset, -2.0, 4 - 5 * index)
            ]
            model.append({'set': corr_3d, 'pattern': 'bru'})

        corr_3d = [(10 * offset, 0.0, 10),
                   (10 * offset, -2.0, 10),
                   (10 * offset, -2.0, 0),
                   (10 * offset, 0.0, 0)]
        model.append({'set': corr_3d, 'pattern': 'rightbuildingfar'})


def contains(element, list):
    try:
        return bool(list.index(element))
    except ValueError:
        return False


def processPolygon(polygon):
    length = len(polygon)
    rows = max([x for (x, y) in polygon])
    columns = max([y for (x, y) in polygon])
    polygon.append((0.0, 0.0))
    codes = [Path.MOVETO]
    for index in range(length - 1):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(polygon, codes)
    points = []
    if mode == 'V':
        for index in range(rows):
            row = [(index, x) for x in range(columns)]
            check = path.contains_points(row)
            temp_points = ([row[i] for i, j in enumerate(check) if j == True and not contains(row[i], polygon)])
            if (len(temp_points) > 0):
                points.append(temp_points)
    else:
        for index in range(columns):
            col = [(x, index) for x in range(rows)]
            check = path.contains_points(col)
            temp_points = ([col[i] for i, j in enumerate(check) if j == True and not contains(col[i], polygon)])
            if (len(temp_points) > 0):
                points.append(temp_points)
    return points


def quantize(max_x, min_x, max_y, min_y, input_list):
    output_list = []
    for (x, y) in input_list:
        output_list.append((int((y - min_y) / (max_y - min_y) * 10000 * (viewport[1] - 1) / 10000),
                            int((x - min_x) / (max_x - min_x) * 10000 * (viewport[0] - 1) / 10000)))
    return output_list


def mapTexture(texture_image, input_points, output_points, output_img):
    transform_matrix = cv2.getPerspectiveTransform(input_points, np.asarray(output_points, dtype='float32'))
    points = processPolygon(output_points)
    inverse_transform = la.inv(transform_matrix)
    max_x = max([x[0] for x in input_points])
    max_y = max([x[1] for x in input_points])
    for row in points:
        for point in row:
            output_matrix = np.matrix([[point[0]], [point[1]], [1]], dtype='float32')
            corr_input_matrix = inverse_transform * output_matrix
            corr_x = int(corr_input_matrix[0] / corr_input_matrix[2])
            corr_y = int(corr_input_matrix[1] / corr_input_matrix[2])
            if corr_y > max_y:
                corr_y = max_y
            if corr_x > max_x:
                corr_x = max_x
            # print texture_image.shape, corr_x, corr_y
            output_img[-point[1]][point[0]] = texture_image[corr_x][corr_y]
    return output_img


def switchOffPixelsInArray(array, points):
    pointsInTexture = np.array(processPolygon(points))
    pointsInTexture = np.reshape(pointsInTexture, (pointsInTexture.shape[0] * pointsInTexture.shape[1], 2))
    newArray = createNullImage(array.shape)
    for point in pointsInTexture:
        newArray[point[0]][point[1]] = array[point[0]][point[1]]
    return newArray


def projectModelPoints():
    global out_img
    camera_position = np.array([0.00, -11.00, 5.00])
    camera_orientation = np.matrix([[0.00, 0.00, 1.00], [1.00, 0.00, 0.00], [0.00, 1.00, 0.00]])
    modelsProjected = []
    allProjectedPoints = []
    for i in range(len(model)):
        single3dSet = model[i]['set']
        singleTexture = np.array(textures[model[i]['pattern']])
        projectedPoints = []

        # cutoff model by camera plane - new points
        temp_points, lines, factors = get_cutoff_points(np.asarray(single3dSet), camera_position, camera_orientation)
        if (len(temp_points) == 0):
            continue
        elif len(temp_points) < 4:
            temp_points, lines, factors = add_dummy_point(temp_points, lines, factors)
        texture_container = np.array([[0, 0], [0, singleTexture.shape[1] - 1],
                                      [singleTexture.shape[0] - 1, singleTexture.shape[1] - 1],
                                      [singleTexture.shape[0] - 1, 0]])
        temp_texture = get_corners_of_cut_texture(texture_container, lines, factors)
        # new texture
        # singleTexture = switchOffPixelsInArray(singleTexture, temp_texture)
        for point in temp_points:
            projectedPoint = return_projected_point(np.array(point), (camera_position - camera_orientation[2].getA1()),
                                                    viewing_angle_in_radians,
                                                    camera_orientation)
            if (projectedPoint is not None):
                projectedPoints.append(projectedPoint)
                allProjectedPoints.append(projectedPoint)
            if len(projectedPoints) >= 4:
                modelsProjected.append(
                    {'set': projectedPoints, 'pattern': singleTexture, 'patternCoords': temp_texture})

                # start projecting on image - quantize, then assign
    min_x = min([x[0] for x in allProjectedPoints])
    min_y = min([x[1] for x in allProjectedPoints])
    max_x = max([x[0] for x in allProjectedPoints])
    max_y = max([x[1] for x in allProjectedPoints])
    for row in modelsProjected:
        input_texture = np.array(row['patternCoords'], dtype=np.float32)
        out_img = mapTexture(row['pattern'], input_texture, quantize(max_x, min_x, max_y, min_y, row['set']), out_img)


viewport = (300, 400)
viewing_angle_in_radians = degtorad(90)
mode = 'V'
model = []
textures = {}

defineModel()
populate_texture_list('textures.csv')
out_img = createNullImage(viewport)
projectModelPoints()
out_img = carryOutDithering(5, out_img)
cv2.imwrite("pers.jpg", out_img)






