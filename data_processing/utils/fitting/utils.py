
import os
import os.path as osp
import yaml
import torch
import sys
import numpy as np
# from utils.data_parser import create_dataset
# from utils.utils import JointMapper, load_camera_para, get_rot_trans
from utils.fitting import smplx
# from camera import create_camera
from utils.prior import create_prior
from utils.fitting.prior import load_vposer
from utils.fitting.umeyama import umeyama
import cv2

def init():

    setting = {}
    # assert cuda is available
    use_cuda =True
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    #read gender
    input_gender = 'neutral'
    model_type = 'smpllsp'
    pose_format = 'lsp14' 
    float_dtype = 'float32'

    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    model_params = dict(model_path='smpl/smpl/SMPL_NEUTRAL.pkl',
                        create_global_orient=True,
                        create_body_pose=False,
                        create_betas=True,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=False,
                        create_reye_pose=False,
                        create_transl=True, #set transl in multi-view task  --Buzhen Huang 07/31/2019
                        create_scale=True,
                        dtype=dtype)

    model = smplx.create_scale(gender=input_gender, **model_params)

    # create prior
    body_pose_prior = create_prior(
        prior_type='l2',
        dtype=dtype)
    shape_prior = create_prior(
        prior_type='l2',
        dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
    else:
        device = torch.device('cpu')
    
    # load vposer
    vposer = None
    pose_embedding = None
    batch_size = 1
    vposer = load_vposer(vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()
    pose_embedding = torch.zeros([batch_size, 32],
                                    dtype=dtype, device=device,
                                    requires_grad=True)

    # return setting
    setting['model'] = model
    setting['dtype'] = dtype
    setting['device'] = device
    setting['vposer'] = vposer
    setting['body_pose_prior'] = body_pose_prior
    setting['shape_prior'] = shape_prior
    setting['pose_embedding'] = pose_embedding
    setting['batch_size'] = batch_size
    return setting


def init_guess(setting, origin_mesh):
    model = setting['model']
    dtype = setting['dtype']
    batch_size = setting['batch_size']
    device = setting['device']
    est_scale = False
    fixed_scale = 1.

    # reset model
    init_t = torch.zeros((1,3), dtype=dtype)
    init_r = torch.zeros((1,3), dtype=dtype)
    init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_shape = torch.zeros((1,10), dtype=dtype)
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)

    init_pose = torch.zeros((1,69), dtype=dtype).cuda()
    model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)

    regressor = model.J_regressor.detach().cpu().numpy()
    init_verts = model_output.vertices[0].detach().cpu().numpy()
    target_verts = origin_mesh

    init_joints = np.dot(regressor, init_verts)[[1,2,16,17]]
    target_joints = np.dot(regressor, target_verts)[[1,2,16,17]]

    # get transformation
    rot, trans, scale = umeyama(init_joints, target_joints, False)
    rot = cv2.Rodrigues(rot)[0]
    # apply to model
    if est_scale:
        init_s = torch.tensor(scale, dtype=dtype)
    else:
        init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_t = torch.tensor(trans, dtype=dtype)
    init_r = torch.tensor(rot, dtype=dtype).reshape(1,3)
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)

    with torch.no_grad():   
        setting['pose_embedding'].fill_(0)

    # visualize
    if False:
        if kwargs.get('use_vposer'):
            vposer = setting['vposer']
            init_pose = vposer.decode(
                setting['pose_embedding'], output_type='aa').view(
                    1, -1)
        else:
            init_pose = torch.zeros((1,69), dtype=dtype).cuda()
        model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
        joints = model_output.joints.detach().cpu().numpy()[0]
        verts = model_output.vertices.detach().cpu().numpy()[0]

        from utils.utils import joint_projection, surface_projection
        for i in range(1):
            joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True)
            surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 5)


