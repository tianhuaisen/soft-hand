# -*- coding: utf-8 -*-
from manopth import manolayer
from utils import func, smoother
from Finger_Main import *
from poseCheck import *
from Fetch_data import *
import numpy as np
import open3d
import multiprocessing
import get_sensor
import torch
import time

ds = DataStore()

def open_hand():
    # Palm model mano root directory
    _mano_root = './hand/manopth/models'
    # Filter
    mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
    view_mat = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0],
                         [0.0, 0, 1.0]])

    # Load mano right hand model
    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side="right",
                               mano_root=_mano_root,
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')

    # Get serial port data, multi-process
    global ds
    new_p1 = multiprocessing.Process(target=Wave_run, args=(ds,))
    new_p1.start()

    # Hand pose parameter generation
    pose, shape = func.initiate("zero")     # pose=48,shape=10
    # print(shape)
    # print(pose0)


    pose_raw = [0,0,0,0,0]
    pose0 = hand_ges(pose_raw)


    # 3D drawing
    mesh = open3d.geometry.TriangleMesh()
    hand_verts, j3d_recon = mano(pose0, shape.float())
    mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
    hand_verts = hand_verts.numpy()[0]
    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    viewer = open3d.visualization.Visualizer()
    viewer.create_window(width=1080, height=810, left=100, top=150, window_name='mesh')
    viewer.add_geometry(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_renderer()

    n = 1
    flag = 0
    ser = get_sensor.openSer()
    # Updated 3D mapping

    while True:

        hand_verts, j3d_recon = mano(pose0, shape.float())
        mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
        hand_verts = hand_verts.numpy()[0]
        hand_verts = mesh_fliter.process(hand_verts)
        hand_verts = np.matmul(view_mat, hand_verts.T).T

        mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
        mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        viewer.update_geometry(mesh)
        viewer.poll_events()
        viewer.update_renderer()

        data = get_sensor.getSerdata(ser)
        if data != None and len(data):
            data = list(map(float, data))
            data = [data[0], data[1]]
            ds.set_data(data)
            pose0 = hand_ges(data)



if __name__ == '__main__':
    open_hand()

