import cv2
import os
import numpy as np
import torch
from utils.projection import surface_projection, joint_projection
from tqdm import tqdm
from copy import deepcopy
from utils.module_utils import vis_img

def viz_annot(dataset_dir, param, smpl_neutral, smpl_male, smpl_female, model_type):
    # visulize
    viz_im = cv2.imread(os.path.join(dataset_dir, param['img_path']))
    h = int(viz_im.shape[0] / 100)
    img_path = param['img_path']
    del param['img_path']
    if 'h_w' in param.keys():
        img_h, img_w = param['h_w']
        del param['h_w']
    else:
        img_h, img_w = viz_im.shape[:2]
        
    for i in param:

        if 'gender' not in param[i].keys():
            smpl = smpl_neutral
        else:
            if param[i]['gender'] == 'male' or param[i]['gender'] == 1:
                smpl = smpl_male
            elif param[i]['gender'] == 'female' or param[i]['gender'] == 0:
                smpl = smpl_female
            else:
                smpl = smpl_neutral

        # viz_im = img #.copy()

        if model_type == 'smpl':
            if param[i]['betas'] is not None:
                if 'extri' in param[i].keys():
                    extri = np.array(param[i]['extri'])
                else:
                    extri = np.eye(4)
                intri = np.array(param[i]['intri'])
                beta = torch.from_numpy(np.array(param[i]['betas'])).reshape(-1, 10).to(torch.float32)
                pose = torch.from_numpy(np.array(param[i]['pose'])).reshape(-1, 72).to(torch.float32)
                trans = torch.from_numpy(np.array(param[i]['trans'])).reshape(-1, 3).to(torch.float32)
                verts, joints = smpl(beta, pose, trans)

                _, viz_im = surface_projection(verts[0].detach().numpy(), smpl.faces, None, extri, intri, viz_im, viz=False)
        elif model_type == 'smplx':
            if param[i]['smplx_betas'] is not None:
                if 'extri' in param[i].keys():
                    extri = np.array(param[i]['extri'])
                else:
                    extri = np.eye(4)
                intri = np.array(param[i]['intri'])
                beta = torch.from_numpy(np.array(param[i]['smplx_betas'])).reshape(-1, 10).to(torch.float32)
                pose = torch.from_numpy(np.array(param[i]['smplx_pose'])).reshape(-1, 72).to(torch.float32)
                trans = torch.from_numpy(np.array(param[i]['smplx_trans'])).reshape(-1, 3).to(torch.float32)
                verts, joints = smpl(beta, pose, trans)

                _, viz_im = surface_projection(verts[0].detach().numpy(), smpl.faces, None, extri, intri, viz_im, viz=False)

        if param[i]['halpe_joints_3d'] is not None:
            if 'extri' in param[i].keys():
                extri = np.array(param[i]['extri'])
            else:
                extri = np.eye(4)
            intri = np.array(param[i]['intri'])
            joints = np.array(param[i]['halpe_joints_3d'])[:,:3]
            joints_2d, _ = joint_projection(joints, extri, intri, viz_im, viz=False)

            for p in joints_2d:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h+3, (0,255,255), -1)

        if param[i]['halpe_joints_2d'] is not None:
            gt_joints = np.array(param[i]['halpe_joints_2d']).reshape(-1, 3)[:,:2]
            for p in gt_joints:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,0,255), -1)

        # if param[i]['halpe_joints_2d_pred'] is not None:
        #     alpha_joints = np.array(param[i]['halpe_joints_2d_pred']).reshape(-1,3)[:,:2]
        #     for p in alpha_joints:
        #         viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,255,0), -1)

        # if param[i]['halpe_joints_2d_det'] is not None:
        #     alpha_joints = np.array(param[i]['halpe_joints_2d_det']).reshape(-1,3)[:,:2]
        #     for p in alpha_joints:
        #         viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (255,0,0), -1)

        # if param[i]['mask_path'] is not None:
        #     mask = cv2.imread(os.path.join(dataset_dir, param[i]['mask_path']), 0)
        #     ratiox = 800/int(mask.shape[0])
        #     ratioy = 800/int(mask.shape[1])
        #     if ratiox < ratioy:
        #         ratio = ratiox
        #     else:
        #         ratio = ratioy
        #     cv2.namedWindow('mask',0)
        #     cv2.resizeWindow('mask',int(mask.shape[1]*ratio),int(mask.shape[0]*ratio))
        #     #cv2.moveWindow(name,0,0)
        #     if mask.max() > 1:
        #         mask = mask/255.
        #     cv2.imshow('mask',mask)
        if param[i]['bbox'] is not None:
            param[i]['bbox'] = np.array(param[i]['bbox']).reshape(2,2).tolist()
            viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['bbox'][0], dtype=np.int)), tuple(np.array(param[i]['bbox'][1], dtype=np.int)), color=(255,255,0), thickness=5)
        # if param[i]['det_bbox'] is not None:
        #     viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['det_bbox'][0], dtype=np.int)), tuple(np.array(param[i]['det_bbox'][1], dtype=np.int)), color=(255,0,255), thickness=5)
        #     pass
    output = os.path.join('output/vis_data', img_path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cv2.imwrite(output, viz_im)
        # vis_img('img', viz_im)

def vis_smpl_3d(annot, smpl_neutral, smpl_male, smpl_female, dataset_dir, set, model_type, output_dir='./output', type='annot'):
    if type == 'annot':
        for i, seq in enumerate(tqdm(annot, total=len(annot))):
            for j, frame in enumerate(seq):
                viz_annot(dataset_dir, deepcopy(frame), smpl_neutral, smpl_male, smpl_female, model_type)
