# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
import scipy.linalg as linalg


# Initialize the pose matrix
def poseIni():

    pose0 = torch.ones([1,16,3,3],dtype=torch.float) * torch.eye(3)

    return pose0

# Changing transformations of the pose matrix
def poseCheck(index, matrix, pose0):

    matrix = torch.FloatTensor(matrix)
    pose0[0][index] = torch.mm(pose0[0][index], matrix)

    return pose0


# Rotation matrix, Euler angles
def rotate_mat(axisinput, radian):
    rot_matrix = linalg.expm(
        np.cross(np.eye(3), axisinput / linalg.norm(axisinput) * radian))
    return rot_matrix



# Index is the number of the joint of the finger(array type).
# Axis has 3 parameters: x,y,z(String type).
# Angle is the input angle in degrees(int type).
def getPose(index, axis, angle, pose0):
    # The x, y and z axes can be customized.
    axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    if axis == 'x':
        axisinput = axis_x
    elif axis == 'y':
        axisinput = axis_y
    elif axis == 'z':
        axisinput = axis_z
    else:
        print("axis input wrong, default use axis_x")
        axisinput = axis_x
    # Angle of rotation
    angle = angle*math.pi/180
    # Return the rotation matrix
    rot_matrix = rotate_mat(axisinput, angle)
    # print(rot_matrix)
    resultpose = poseCheck(index, rot_matrix, pose0)
    return resultpose



# Compound rotation angle, need to enter the angle of xyz three axes in sequence
def getPoseMul(index, angle1,angle2,angle3, pose0):

    pose0 = getPose(index, 'x', angle1, pose0)
    pose0 = getPose(index, 'y', angle2, pose0)
    pose0 = getPose(index, 'z', angle3, pose0)

    return pose0

def thumb(pose0, thu):
    pose0 = getPoseMul(13, 30/90*thu, 15/90*thu, 0, pose0)
    pose0 = getPoseMul(14, 30/90*thu, -25/90*thu, 0, pose0)
    pose0 = getPoseMul(15, 45/90*thu, 0, 60/90*thu, pose0)
    return pose0

def index(pose0, ind):
    pose0 = getPoseMul(1, 0, 0, 65/90*ind, pose0)
    pose0 = getPoseMul(2, 0, 0, 80/90*ind, pose0)
    pose0 = getPoseMul(3, 0, 0, 70/90*ind, pose0)
    return pose0

def middle(pose0, mid):
    pose0 = getPoseMul(4, -5/90*mid, 0, 75/90*mid, pose0)
    pose0 = getPoseMul(5, -15/90*mid, -10/90*mid, 75/90*mid, pose0)
    pose0 = getPoseMul(6, -10/90*mid, 0, 65/90*mid, pose0)
    return pose0

def ring(pose0, rin):
    pose0 = getPoseMul(10, -5/90*rin, 0, 65/90*rin, pose0)
    pose0 = getPoseMul(11, -30/90*rin, -10/90*rin, 80/90*rin, pose0)
    pose0 = getPoseMul(12, -15/90*rin, 0, 70/90*rin, pose0)
    return pose0

def pinky(pose0, pin):
    pose0 = getPoseMul(7, -30/90*pin, -10/90*pin, 50/90*pin, pose0)
    pose0 = getPoseMul(8, -30/90*pin, 0, 80/90*pin, pose0)
    pose0 = getPoseMul(9, -60/90*pin, 0, 60/90*pin, pose0)
    return pose0

def hand_ges(angel):
    pose0 = poseIni()
    pose0 = thumb(pose0, 45)
    pose0 = index(pose0, angel[0])
    pose0 = middle(pose0, angel[1])
    pose0 = ring(pose0, 45)
    pose0 = pinky(pose0, 45)
    return pose0

if __name__ == '__main__':
    a = 0
    while True:
        a = a+1
        if a <= 90:
            pose_raw = [a, a, a, a, a]
            pose0 = hand_ges(pose_raw)
            print(pose0)
            print(a)
        else:
            a = 0
