import numpy as np
import cv2
import math
import time

from projectTextures import projectModelPoints, defineModel, populate_texture_list
from projection import rotate, rotation_quaternion, quat2rot

orig_orientation = np.matrix([[0.00, 0.00, 1.00], [1.00, 0.00, 0.00], [0.00, 1.00, 0.00]])
orig_position = np.array([-4, -5, 1], dtype='float32')
model = []
textures = {}

out_name = "output%d.mov" % int(time.time())
out = cv2.VideoWriter(out_name, cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 25.0, (800, 600), 1)

model = defineModel(model)
textures = populate_texture_list('textures.csv', textures)

# Turn left for 2 sec
# rot_angle = 0
# for i in range(0, 30):
# 	print camera_orientation
# 	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
# 	out_img = np.array(out_img, dtype='uint8')
# 	out.write(out_img)
# 	rot_angle += 3;
# 	camera_orientation *= quat2rot(rotation_quaternion([0, 0, 1], rot_angle))
#Look right for 2 sec

# angle_step = 3
# orig_orientation = camera_orientation
# for i in range(0, 30):
# 	print camera_orientation
# 	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
# 	out_img = np.array(out_img, dtype='uint8')
# 	out.write(out_img);
# 	orig_orientation *= quat2rot(rotation_quaternion([0, 0, 1], angle_step))

# running forward and turning 90 degress right for 6 sec
camera_position = orig_position
camera_orientation = orig_orientation
angle_step = 90.0 / 150.0
for i in range(0, 150):
    out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
    out_img = np.array(out_img, dtype='uint8')
    # img_name = "frame%d.png" % i
    # cv2.imwrite(img_name, out_img)
    out.write(out_img)
    camera_position[1] += (math.pi) / 90
    camera_position[2] = (math.cos((i * 2 * math.pi) / 30) / 2)
    print i * angle_step
    camera_orientation = orig_orientation * quat2rot(rotation_quaternion([0, 0, 1], i * angle_step))
    print camera_position
    print camera_orientation

#Roll in place for 1 sec
# angle_step = 6
# for i in range(0, 60):
# 	print camera_orientation
# 	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
# 	out_img = np.array(out_img, dtype='uint8')
# 	out.write(out_img)
# 	camera_orientation *= quat2rot(rotation_quaternion([0,1,0], angle_step))

# print camera_orientation
out.release()

# for i in range(0,30):
# 	img_name = "frame%d.png" % i
# 	in_img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_COLOR)
# 	out.write(in_img)

