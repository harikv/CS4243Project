import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    if length(v1) * length(v2) == 0:
        return 0
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def quatmult(q1, q2):
    # quaternion multiplication
    out = [0, 0, 0, 0]
    out[0] = (q1[0] * q2[0]) - (q1[1] * q2[1]) - (q1[2] * q2[2]) - (q1[3] * q2[3])
    out[1] = (q1[0] * q2[1]) + (q1[1] * q2[0]) + (q1[2] * q2[3]) - (q1[3] * q2[2])
    out[2] = (q1[0] * q2[2]) - (q1[1] * q2[3]) + (q1[2] * q2[0]) + (q1[3] * q2[1])
    out[3] = (q1[0] * q2[3]) + (q1[1] * q2[2]) - (q2[2] * q2[1]) + (q1[3] * q2[0])
    return out


def degtorad(deg):
    # Converts degrees to radians
    rad = (math.pi / 180) * deg
    return rad


def rotation_quaternion(axis, angle):
    # axis is rotation unit vector axis
    #angle is angle of rotation in degrees
    rotation_quat = [0, 0, 0, 0]
    angle_rad = degtorad(angle / 2)
    rotation_quat[0] = math.cos(angle_rad)
    rotation_quat[1] = math.sin(angle_rad) * axis[0]
    rotation_quat[2] = math.sin(angle_rad) * axis[1]
    rotation_quat[3] = math.sin(angle_rad) * axis[2]
    return rotation_quat


def rotate(point, axis, angle):
    # input is a input 3d point
    #axis is the rotation unit vector axis
    #angle is angle of rotation in degrees
    input_quat = [0, 0, 0, 0]
    input_quat[1:] = point
    rotation_quat = rotation_quaternion(axis, angle)
    rotation_conj = [0, 0, 0, 0]
    rotation_conj[0] = rotation_quat[0]
    rotation_conj[1] = -rotation_quat[1]
    rotation_conj[2] = -rotation_quat[2]
    rotation_conj[3] = -rotation_quat[3]
    rotated_point = quatmult(quatmult(rotation_quat, input_quat), rotation_conj)
    return rotated_point


def quat2rot(q):
    """
    Translates a quarternion into a rotation matrix
    :param q: rotation quarternion
    :return: rotational matrix corresponding to input quarternion
    """
    rot_matrix = np.zeros([3, 3])
    rot_matrix[0][0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    rot_matrix[0][1] = 2 * (q[1] * q[2] - q[0] * q[3])
    rot_matrix[0][2] = 2 * (q[1] * q[3] + q[0] * q[2])
    rot_matrix[1][0] = 2 * (q[1] * q[2] + q[0] * q[3])
    rot_matrix[1][1] = q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2
    rot_matrix[1][2] = 2 * (q[2] * q[3] - q[0] * q[1])
    rot_matrix[2][0] = 2 * (q[1] * q[3] - q[0] * q[2])
    rot_matrix[2][1] = 2 * (q[2] * q[3] + q[0] * q[1])
    rot_matrix[2][2] = q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2
    return np.matrix(rot_matrix)


def orthographic_proj(point, camera_position, camera_axes):
    point_camera_frame = np.matrix(point - camera_position)
    camera_x_axis = np.matrix(camera_axes[0]).transpose()
    camera_y_axis = np.matrix(camera_axes[1]).transpose()
    projected_point = [0, 0]
    projected_point[0] = (point_camera_frame * camera_x_axis).item(0)
    projected_point[1] = (point_camera_frame * camera_y_axis).item(0)
    return projected_point


def checkPointBehind(point, camera_position, optical_axis, viewing_angle):
    vector_point_camera = point - camera_position
    in_angle = angle(vector_point_camera, optical_axis.getA()[0])
    return in_angle < viewing_angle


def perspective_proj(point, camera_position, camera_axes, viewing_angle):
    point_camera_frame = np.matrix(point - camera_position)
    if not checkPointBehind(point, camera_position, camera_axes[2], viewing_angle):
        return None
    camera_x_axis = np.matrix(camera_axes[0]).transpose()
    camera_y_axis = np.matrix(camera_axes[1]).transpose()
    camera_optical_axis = np.matrix(camera_axes[2]).transpose()
    projected_point = [0, 0]
    try:
        projected_point[0] = ((point_camera_frame * camera_x_axis).item(0)) / (
        (point_camera_frame * camera_optical_axis).item(0))
        projected_point[1] = ((point_camera_frame * camera_y_axis).item(0)) / (
        (point_camera_frame * camera_optical_axis).item(0))
    except ZeroDivisionError:
        return None
    return projected_point


def return_projected_point(point, cam_pos, viewing_angle, camera_orientation):
    return perspective_proj(point, cam_pos, camera_orientation, viewing_angle)