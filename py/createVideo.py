import numpy as np
import cv2
import math
import time

from projectTextures import projectModelPoints, defineModel, populate_texture_list
from projection import rotate, rotation_quaternion, quat2rot

camera_orientation = np.matrix([[0.00, 0.00, 1.00], [1.00, 0.00, 0.00], [0.00, 1.00, 0.00]])
camera_position = np.array([0, -8, 1], dtype='float32')
model = []
textures = {}

out_name = "output%d.mov" % int(time.time())
out = cv2.VideoWriter(out_name,cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 15.0, (400,300), 1)

model = defineModel(model)
textures = populate_texture_list('textures.csv', textures)

#Turn left for 2 sec
# rot_angle = 0
# for i in range(0, 30):
# 	print camera_orientation
# 	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
# 	out_img = np.array(out_img, dtype='uint8')
# 	out.write(out_img)
# 	rot_angle += 3;
# 	camera_orientation *= quat2rot(rotation_quaternion([0, 0, 1], rot_angle))
#Look right for 2 sec

# rot_angle = 0
# orig_orientation = camera_orientation
# for i in range(0, 30):
# 	print camera_orientation
# 	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
# 	out_img = np.array(out_img, dtype='uint8')
# 	out.write(out_img)
# 	rot_angle -= 1;
# 	orig_orientation *= quat2rot(rotation_quaternion([0, 0, 1], rot_angle))

#running forward for 5 sec
for i in range(0, 75):
	print camera_position
	out_img = projectModelPoints(camera_position, camera_orientation, model, textures)
	out_img = np.array(out_img, dtype='uint8')
	# img_name = "frame%d.png" % i 
	# cv2.imwrite(img_name, out_img)
	out.write(out_img)
	camera_position[1] += (math.pi)/30
	camera_position[2] = (math.sin((i*4*math.pi)/30)/2)

#Roll in place for 1 sec

out.release()

# for i in range(0,30):
# 	img_name = "frame%d.png" % i
# 	in_img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_COLOR)
# 	out.write(in_img)

