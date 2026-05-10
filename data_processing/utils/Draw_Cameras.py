import numpy as np
import sys

from numpy import linalg
from numpy.lib.index_tricks import AxisConcatenator
sys.path.append('./')
from utils.module_utils import load_camera_para, rigid_transform_3D, save_camparam
from utils.Visualization3D import Visualization
from utils.umeyama import umeyama
import os
import torch
import cv2
import torch.optim as optim
from torchgeometry import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix

def add_camera_mesh(extrinsic, camerascale=1):

    # 12 points camera
    r = np.zeros((3,4,3))
    r[0][0] = np.array([-0.5, 0.5, 0]) * camerascale
    r[0][1] = np.array([0.5, 0.5, 0]) * camerascale
    r[0][2] = np.array([0.5, -0.5, 0]) * camerascale
    r[0][3] = np.array([-0.5, -0.5, 0]) * camerascale

    r[1][0] = np.array([-1, 1, 1]) * camerascale
    r[1][1] = np.array([1, 1, 1]) * camerascale
    r[1][2] = np.array([1, -1, 1]) * camerascale
    r[1][3] = np.array([-1, -1, 1]) * camerascale

    r[2][0] = np.array([-0.5, 0.5, -2]) * camerascale
    r[2][1] = np.array([0.5, 0.5, -2]) * camerascale
    r[2][2] = np.array([0.5, -0.5, -2]) * camerascale
    r[2][3] = np.array([-0.5, -0.5, -2]) * camerascale

    P = np.zeros((3, 40))
    for i in range(3):
        P[:,i * 8 + 0] = r[i][0] 
        P[:,i * 8 + 1] = r[i][1]
        P[:,i * 8 + 2] = r[i][1] 
        P[:,i * 8 + 3] = r[i][2]
        P[:,i * 8 + 4] = r[i][2] 
        P[:,i * 8 + 5] = r[i][3]
        P[:,i * 8 + 6] = r[i][3] 
        P[:,i * 8 + 7] = r[i][0]

    for i in range(2):
        P[:,24 + i * 8 + 0] = r[0][0] 
        P[:,24 + i * 8 + 1] = r[i + 1][0]
        P[:,24 + i * 8 + 2] = r[0][1] 
        P[:,24 + i * 8 + 3] = r[i + 1][1]
        P[:,24 + i * 8 + 4] = r[0][2] 
        P[:,24 + i * 8 + 5] = r[i + 1][2]
        P[:,24 + i * 8 + 6] = r[0][3] 
        P[:,24 + i * 8 + 7] = r[i + 1][3]

    # // transform from camera space to object space
    # // this step is critical for visualizing the cameras since our viewpoint is in the object space
    M = np.linalg.inv(extrinsic)
    for i in range(P.shape[1]):
        t = np.ones((4,))
        t[:3] = P[:,i]
        p = np.dot(M, t)
        P[:,i] = p[:3] / p[3]

    return P

def align_camera_nonlinear(pred_cams, gt_cams, est_scale=False, viz=False):

    # # visualize input cameras
    # viz = Visualization()
    # for cam in gt_cams:
    #     t = add_camera_mesh(cam)
    #     viz.visualize_cameras(t.T, [1, 0.5, 0.5])

    # for cam in pred_cams:
    #     t = add_camera_mesh(cam)
    #     viz.visualize_cameras(t.T, [0.5, 0.5, 1])

    gt_pos = []
    pred_pos = []
    for i, (cam_gt, cam_pre) in enumerate(zip(gt_cams, pred_cams)):
        pos = np.dot(np.linalg.inv(cam_gt), np.array([0,0,0,1]))[:3]
        gt_pos.append(pos)
        pos = np.dot(np.linalg.inv(cam_pre), np.array([0,0,0,1]))[:3]
        pred_pos.append(pos)
    gt_pos = np.array(gt_pos)
    pred_pos = np.array(pred_pos)

    R_, t_, flag = rigid_transform_3D(pred_pos, gt_pos)
    R_ = cv2.Rodrigues(R_)[0]

    refine_R = torch.tensor(R_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
    refine_t = torch.tensor(t_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
    refine_s = torch.tensor([1.0], device=torch.device('cuda'), dtype=torch.float64, requires_grad=True)
    pred_pos = torch.tensor(pred_pos, device=torch.device('cuda'))
    gt_pos = torch.tensor(gt_pos, device=torch.device('cuda'))

    if est_scale:
        final_param = [refine_R, refine_t, refine_s]
    else:
        final_param = [refine_R, refine_t]

    optimizer = optim.Adam(filter(lambda p:p.requires_grad, final_param), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    def loss_func(pred_R, pred_t, pred_s, init, gt_pos):
        rot_mat = angle_axis_to_rotation_matrix(pred_R)
        rot_mat = rot_mat[0]
        rot_mat[:3,3] = pred_t
        ones = torch.ones((init.shape[0], 1), dtype=init.dtype, device=init.device)
        init = init * pred_s
        init = torch.cat([init, ones], dim=1).permute(1,0)
        pos = torch.matmul(rot_mat, init).permute(1,0)
        pos = pos[:,:3]
        loss = torch.norm(pos - gt_pos, dim=1).sum()
        return loss

    loss_init = loss_func(refine_R, refine_t, refine_s, pred_pos, gt_pos) + 1
    while True:

        loss = loss_func(refine_R, refine_t, refine_s, pred_pos, gt_pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        if torch.abs(loss - loss_init) < 1e-5:
            break
        # print(loss)
        loss_init = loss

    transform = angle_axis_to_rotation_matrix(refine_R)[0].detach().cpu().numpy()
    transform[:3,3] = refine_t.detach().cpu().numpy()
    s = refine_s.detach().cpu().numpy()
    pred_pos = pred_pos.detach().cpu().numpy() * s
    gt_pos = gt_pos.detach().cpu().numpy()
    out_pos = (np.matmul(transform, np.insert(pred_pos, 3, 1, axis=1).T).T)[:,:3]

    rots = []
    for i, cam in enumerate(pred_cams):
        cam[:3,3] = cam[:3,3] * s
        transform1 = np.linalg.inv(transform)
        rot = np.dot(cam, transform1)
        rots.append(rot)

    if viz:
        visualize = Visualization()
        visualize.visualize_points(gt_pos, [1,0,0])
        visualize.visualize_points(out_pos, [0,0,1])
        for cam in gt_cams:
            t = add_camera_mesh(cam, camerascale=0.1)
            visualize.visualize_cameras(t.T, [1, 0.5, 0.5])

        for cam in rots:
            t = add_camera_mesh(cam, camerascale=0.1)
            visualize.visualize_cameras(t.T, [0, 0.5, 1])
        while True:
            visualize.show()

    rots_errors = []
    for pred_c, gt_c in zip(rots, gt_cams):
        p_rot = cv2.Rodrigues(pred_c[:3,:3])[0].reshape(1, -1)
        g_rot = cv2.Rodrigues(gt_c[:3,:3])[0].reshape(1, -1)
        error = np.linalg.norm(abs(p_rot) - abs(g_rot))
        rots_errors.append(error)

    rot_error = np.mean(np.array(rots_errors))
    pos_error = np.mean(np.linalg.norm(out_pos - gt_pos, axis=0))

    return transform, s, rot_error, pos_error, rots

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    flag = 0
    if np.linalg.det(R) < 0:
        #print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)
        flag = 1

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t, flag

def save_mesh(path):
    f = open(path, 'r')
    lines = f.readlines()
    verts, faces = [], []
    for l in lines:
        l = l.rstrip('\n')
        l = l.split(' ')
        if l[0] == 'v':
            verts.append([float(l[1]), float(l[2]),float(l[3]),])
        elif l[0] == 'f':
            try:
                faces.append([int(l[1]), int(l[2]),int(l[3]),])
            except:
                faces.append([int(l[1].split('//')[0]), int(l[2].split('//')[0]),int(l[3].split('//')[0]),])
    return np.array(verts), np.array(faces) - 1

def load_obj(path):
    f = open(path, 'r')
    lines = f.readlines()
    verts, faces, colors = [], [], []
    for l in lines:
        l = l.rstrip('\n')
        l = l.split(' ')
        if l[0] == 'v':
            verts.append([float(l[1]), float(l[2]),float(l[3]),])
            # colors.append([float(l[4]), float(l[5]),float(l[6]),])
        elif l[0] == 'f':
            try:
                faces.append([int(l[1]), int(l[2]),int(l[3]),])
            except:
                faces.append([int(l[1].split('//')[0]), int(l[2].split('//')[0]),int(l[3].split('//')[0]),])
    return np.array(verts), np.array(faces) - 1, np.array(colors)

def write_obj(verts, faces, file_name):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

if __name__ == '__main__':

    # Demo show cameras
    if True:
        visualizer = Visualization()
        extris, intris, _ = load_camera_para(R'D:\HuangBuzhen_Programs\MvSeqRec\Reconstruction\output\cameras\alphapose\00999.txt')
        for cam in extris:
            # if cam[0][0] == 1:
            #     continue
            cam = add_camera_mesh(cam, camerascale=0.1)
            visualizer.visualize_cameras(cam.T, [0,0,1])
        while True:
            visualizer.show()
    else:
        # visualizer = Visualization()
        extris_pred, intris_pred, _ = load_camera_para(r'E:\Results_3DV2021\Ours_cam1\cameras\Panoptic\camparams.txt')
        extris_gt, intris_gt, _ = load_camera_para(r'E:\Render_3DV\ours_panoptic\camparams\Panoptic\camparams.txt')

        verts0, faces, colors = load_obj(r'E:\Results_3DV2021\Ours_cam1\meshes\Panoptic\00650_00.obj')
        verts1, faces, colors = load_obj(r'E:\Results_3DV2021\Ours_cam1\meshes\Panoptic\00650_01.obj')
        verts2, faces, colors = load_obj(r'E:\Results_3DV2021\Ours_cam1\meshes\Panoptic\00650_02.obj')
        verts3, faces, colors = load_obj(r'E:\Results_3DV2021\Ours_cam1\meshes\Panoptic\00650_03.obj')
        verts4, faces, colors = load_obj(r'E:\Results_3DV2021\Ours_cam1\meshes\Panoptic\00650_04.obj')

        verts = np.concatenate((verts0, verts1, verts2, verts3, verts4), axis=0)

        verts, extris_pred = align_camera_nonlinear(extris_pred, extris_gt, verts)

        save_camparam(r'E:\Results_3DV2021\Ours_cam1\output\aligned_cam.txt', intris_pred, extris_pred)

        verts = verts.reshape(-1, 6890, 3)
        for i, vert in enumerate(verts):
            write_obj(vert, faces, os.path.join(r'E:\Results_3DV2021\Ours_cam1\output', '00650_%02d.obj' %i))
        # write_obj(colors, faces, r'E:\Results_3DV2021\PhotoScan\color.obj')

        # gt_pos = []
        # pred_pos = []
        # for cam_gt, cam_pred in zip(extris_gt, extris_pred):
        #     if cam_pred[0][0] == 1:
        #         continue
        #     pos = np.dot(np.linalg.inv(cam_gt), np.array([0,0,0,1]))[:3]
        #     gt_pos.append(pos)
        #     pos = np.dot(np.linalg.inv(cam_pred), np.array([0,0,0,1]))[:3]
        #     pred_pos.append(pos)
        # gt_pos = np.array(gt_pos)
        # pred_pos = np.array(pred_pos)

        # rot, trans, scale = umeyama(gt_pos, pred_pos, True)

        # # v_mean = np.mean(verts)
        # # verts = (verts - v_mean) * scale + v_mean
        # # mean = np.mean(pred_pos, axis=0)
        # # pred_pos = (pred_pos - mean)  * scale + mean

        # # verts = verts * scale
        # # pred_pos = pred_pos * scale

        # transform = np.eye(4)
        # transform[:3,:3] = rot
        # transform[:3,3] = trans

        # # transform[0][0] = transform[0][0] * scale
        # # transform[1][1] = transform[1][1] * scale
        # # transform[2][2] = transform[2][2] * scale

        # # out_pos = (np.matmul(transform, np.insert(pred_pos, 3, 1, axis=1).T).T)[:,:3]
        # # verts = (np.matmul(transform, np.insert(verts, 3, 1, axis=1).T).T)[:,:3]
        
        # rots = []
        # for cam in extris_gt:
        #     rot = np.dot(cam, np.linalg.inv(transform))
        #     rots.append(rot)


        # # visualizer.visualize_points(gt_pos, [1,0,0])
        # # visualizer.visualize_points(pred_pos, [0,0,1])
        # visualizer.visualize_points(verts, [1,1,0])

        

        # for i, cam in enumerate(extris_pred):
        #     if cam[0][0] == 1:
        #         continue
        #     cam = add_camera_mesh(cam, camerascale=0.1)
        #     visualizer.visualize_cameras(cam.T, [0,0,1])


        # for cam in rots:
        #     if cam[0][0] == 1:
        #         continue
        #     cam = add_camera_mesh(cam, camerascale=0.1)
        #     visualizer.visualize_cameras(cam.T, [1,0,0])

        # while True:
        #     visualizer.show()

