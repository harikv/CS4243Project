import cv2
import numpy as np
from matplotlib.path import Path
import csv

textures = {}

# --------------------- writing textures ----------------

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

def getDivisions(pt1, pt2, granularity):
    vector = np.array([(x[1] - x[0]) for x in zip(pt1, pt2)])
    return [(np.array(pt1) + (vector * (i) / (granularity+1))) for i in range(1, granularity + 1)]

def expandTexture(texture, granularity = 1):
    newTexture = []
    for row in range(len(texture)):
        newTexture.append(texture[row])
        if row != len(texture) - 1:
            newTexture.extend(getExtraColors(texture[row], texture[row+1], granularity))
    newTextureT = np.transpose(newTexture, [1,0,2])
    newTexture = []
    for row in range(len(newTextureT)):
        newTexture.append(newTextureT[row])
        if row != len(newTextureT) - 1:
            newTexture.extend(getExtraColors(newTextureT[row], newTextureT[row+1], granularity))
    return np.transpose(newTexture, [1,0,2])

def getExtraColors(row1, row2, granularity):
    """
    Interpolating color rows between two rows
    """
    if len(row1) != len(row2) :
        print "Rows Not Equal"
        return []
    if granularity < 1:
        return []
    color_array = [getDivisions(np.asarray(color1, dtype='float64'), np.asarray(color2, dtype='float64'), granularity) for
                   color1, color2 in zip(row1, row2)]
    return np.transpose(color_array, [1,0,2])

def processPolygon(polygon, rows, columns, mode):
    """
    Finds the points within a particular polygon
    """
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


def buildTextureFromImage(imgURL, pts):
    """
    build an image texture from an image, by specifying the polygon on the image
    """
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
    """
    build an image texture from a solid color
    """
    # colour code is a tuple of rgb values
    new_img = []
    for row in range(height):
        new_img.append([])
        for column in range(width):
            new_img[row].append(np.array(list(colour_code)))
    return np.asarray(new_img)

"""
Started Creation of textures
"""

# front building
textures['front'] = expandTexture(buildTextureFromImage('../project.jpeg', [[706, 777], [836, 777], [836, 870], [706, 870]]), 5)
textures['front-roof'] = expandTexture(buildTextureFromImage('../project.jpeg', [[763, 743], [783, 743], [843, 777], [701, 777]]), 5)
# #sky
textures['sky1'] = buildTextureFromImage('../sky.jpg', [(200, 0), (600, 0), (600, 200), (200, 200)])
textures['sky_main'] = buildTextureFromImage('../sky.jpg', [(200, 200), (600, 200), (600, 600), (200, 600)])
textures['sky3'] = buildTextureFromImage('../sky.jpg', [(200, 600), (600, 600), (600, 800), (200, 800)])
textures['sky4'] = buildTextureFromImage('../sky.jpg', [(0, 200), (200, 200), (200, 600), (0, 600)])
textures['sky5'] = buildTextureFromImage('../sky.jpg', [(600, 200), (800, 200), (800, 600), (600, 600)])

#ground
textures['ground'] = buildSolidTexture(600, 600, (179, 222, 245))

#lawn
textures['lawn'] = expandTexture(buildTextureFromImage('../project.jpeg', [[551, 867], [965, 872], [1636, 1224], [0, 1224]]), 1)

# building-right-unit
textures['bru'] = expandTexture(buildTextureFromImage('../project.jpeg', [[1510, 593], [1572, 568], [1572, 722], [1510, 730]]), 5)

#building roof
textures['roof'] = expandTexture(buildTextureFromImage('../project.jpeg', [[1261, 654], [1323, 655], [1327, 675], [1225, 675]]), 5)

#circular tower
textures['tower'] = expandTexture(buildTextureFromImage('../project.jpeg', [[749, 669], [799, 669], [799, 733], [749, 733]]), 1)

#far tower
textures['ftower'] = expandTexture(buildTextureFromImage('../project.jpeg', [[865, 665], [941, 669], [947, 787], [863, 779]]), 5)

# trees
textures['trees'] = expandTexture(buildTextureFromImage('../project.jpeg', [[1, 327], [583, 633], [537, 863], [3, 901]]), 1)

# #staircase
textures['staircase'] = buildTextureFromImage('../project.jpeg', [[984, 749], [1100, 749], [1100, 849], [984, 849]])

# #corridor
textures['corridor'] = buildTextureFromImage('../project.jpeg', [[836, 787], [941, 787], [941, 859], [836, 859]])

# #rightbuildingfar
textures['rightbuildingfar'] = buildTextureFromImage('../project.jpeg', [[1129, 725], [1172, 725], [1172, 859], [1129, 859]])

#writing textures to csv file
write_texture_file('textures.csv')
