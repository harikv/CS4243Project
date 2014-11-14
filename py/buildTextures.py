import cv2
import numpy as np
from matplotlib.path import Path
import csv

textures = {}

# --------------------- reading and writing textures ----------------

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


def write_texture_file(fileName):
    with open(fileName, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([len(textures)])
        for name, texture in textures.iteritems():
            writer.writerow([name, len(texture[0]), len(texture)])
            for row in texture:
                writer.writerow([[x for x in r] for r in row])


# -----------------making textures-------------------

def contains(element, list):
    try:
        return bool(list.index(element))
    except ValueError:
        return False


def processPolygon(polygon, rows, columns, mode):
    length = len(polygon)
    polygon.append((0.0, 0.0))
    codes = [Path.MOVETO]
    for index in range(length - 1):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(polygon, codes)
    points = []
    if mode == 'V':
        for index in range(rows):
            row = [(x, index) for x in range(columns)]
            check = path.contains_points(row)
            temp_points = ([row[i] for i, j in enumerate(check) if j == True and not contains(row[i], polygon)])
            if (len(temp_points) > 0):
                points.append(temp_points)
    else:
        for index in range(columns):
            col = [(index, x) for x in range(rows)]
            check = path.contains_points(col)
            temp_points = ([col[i] for i, j in enumerate(check) if j == True and not contains(col[i], polygon)])
            if (len(temp_points) > 0):
                points.append(temp_points)
    return points


def buildTextureFromImage(imgURL, pts, mode):
    img = cv2.imread(imgURL, cv2.CV_LOAD_IMAGE_COLOR)
    if(len(img) == 0):
        return
    output_width = max([pts[1][0] - pts[0][0], pts[2][0] - pts[3][0]])
    output_height = max([pts[3][1] - pts[0][1], pts[2][1] - pts[1][1]])
    output_boundaries = [[0,0], [0, output_width], [output_height, output_width], [output_height, 0]]
    transform_matrix = cv2.getPerspectiveTransform(np.asarray(pts, dtype='float32'), np.asarray(output_boundaries, dtype='float32'))
    output_pattern = cv2.warpPerspective(img, transform_matrix, (output_height, output_width))
    return np.transpose(output_pattern, [1,0,2])


def buildSolidTexture(width, height, colour_code):
    # colour code is a tuple of rgb values
    new_img = []
    for row in range(height):
        new_img.append([])
        for column in range(width):
            new_img[row].append(np.array(list(colour_code)))
    return np.asarray(new_img)

# front building
textures['front'] = buildTextureFromImage('../project.jpeg', [[706, 740], [836, 740], [836, 870], [706, 870]], "H")

# #sky
textures['sky1'] = buildTextureFromImage('../sky.jpg', [(200, 0), (600, 0), (600, 200), (200, 200)], "V")
textures['sky_main'] = buildTextureFromImage('../sky.jpg', [(200, 200), (600, 200), (600, 600), (200, 600)], "V")
textures['sky3'] = buildTextureFromImage('../sky.jpg', [(200, 600), (600, 600), (600, 800), (200, 800)], "V")
textures['sky4'] = buildTextureFromImage('../sky.jpg', [(0, 200), (200, 200), (200, 600), (0, 600)], "V")
textures['sky5'] = buildTextureFromImage('../sky.jpg', [(600, 200), (800, 200), (800, 600), (600, 600)], "V")

#ground
textures['ground'] = buildSolidTexture(600, 600, (179, 222, 245))

#lawn
textures['lawn'] = buildTextureFromImage('../project.jpeg', [[551, 867], [965, 872], [1636, 1224], [0, 1224]], "V")

# # building-right-unit
textures['bru'] = buildTextureFromImage('../project.jpeg', [[1510, 593], [1572, 568], [1572, 722], [1510, 730]], "H")

# #staircase
textures['staircase'] = buildTextureFromImage('../project.jpeg', [[984, 749], [1100, 749], [1100, 849], [984, 849]],
                                              "V")

# #corridor
textures['corridor'] = buildTextureFromImage('../project.jpeg', [[836, 787], [941, 787], [941, 859], [836, 859]], "V")

# #rightbuildingfar
textures['rightbuildingfar'] = buildTextureFromImage('../project.jpeg',
                                                     [[1129, 725], [1172, 725], [1172, 859], [1129, 859]], "V")

write_texture_file('textures.csv')
