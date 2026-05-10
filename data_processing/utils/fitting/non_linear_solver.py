'''
 @FileName    : non_linear_solver.py
 @EditTime    : 2021-09-19 21:48:01
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from utils.optimizers import optim_factory

from utils.fitting import fitting

def non_linear_solver(
                    setting,
                    data,
                    batch_size=1,
                    data_weights=[1,1], #[1,1,1,1]
                    body_pose_prior_weights=[4.04e2, 1.78e0], #[4.04e2, 2.04e2, 27.4e0, 1.78e0]
                    shape_weights=[1.0e2, 0.5e1], #[1.0e2, 5.0e1, 1.0e1, 0.5e1]
                    rho=100,
                    loss_type='smplify',
                    visualize=False,
                    use_vposer=True,
                    interactive=True,
                    use_cuda=True,
                    is_seq=False,
                    **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = setting['device']
    dtype = setting['dtype']
    vposer = setting['vposer']
    origin_mesh = data
    model = setting['model']
    pose_embedding = setting['pose_embedding']

    assert (len(data_weights) ==
            len(body_pose_prior_weights) and len(shape_weights) ==
            len(body_pose_prior_weights)), "Number of weight must match"
    
    # process keypoints
    origin_mesh = torch.tensor(origin_mesh, dtype=dtype).to(device=device)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}

    # get weights for each stage
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # create fitting loss
    loss = fitting.create_loss(loss_type=loss_type,
                               rho=rho,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=setting['body_pose_prior'],
                               shape_prior=setting['shape_prior'],
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs)

    data_weight = 1

    # we do not change rotation in multi-view task
    orientations = [model.global_orient]
    results = []

    # Step 1: Optimize the full model
    final_loss_val = 0
    opt_start = time.time()

    for opt_idx, curr_weights in enumerate(opt_weights):

        body_params = list(model.parameters())

        final_params = list(
            filter(lambda x: x.requires_grad, body_params))

        if vposer is not None:
            final_params.append(pose_embedding)

        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params,
            **kwargs)
        body_optimizer.zero_grad()

        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        loss.reset_loss_weights(curr_weights)

        closure = monitor.create_fitting_closure(
            body_optimizer, model,
            origin_mesh=origin_mesh,
            loss=loss, create_graph=body_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_verts=True, return_full_pose=True)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            model,
            pose_embedding=pose_embedding, vposer=vposer,
            use_vposer=use_vposer)

        # if interactive:
        #     if use_cuda and torch.cuda.is_available():
        #         torch.cuda.synchronize()
        #     elapsed = time.time() - stage_start
        #     if interactive:
        #         tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
        #             opt_idx, elapsed))

    # if interactive:
    #     if use_cuda and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #     elapsed = time.time() - opt_start
    #     tqdm.write(
    #         'Body fitting done after {:.4f} seconds'.format(elapsed))
    #     tqdm.write('Body final loss val = {:.5f}'.format(
    #         final_loss_val))


        body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
        # body_pose[:,18:24] = 0.
        # body_pose[:,27:33] = 0.
        # body_pose[:,57:] = 0.

        model_output = model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()


        # Get the result of the fitting process
        result = {key: val.detach().cpu().numpy()
                        for key, val in model.named_parameters()}
        result['pose'] = np.hstack((result['global_orient'], body_pose.detach().cpu().numpy()))
        result['loss'] = final_loss_val
        result['pose_embedding'] = pose_embedding
    return result, vertices
