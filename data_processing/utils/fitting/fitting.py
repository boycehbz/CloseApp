# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

# from utils import utils
import cv2
from model_transfer.utils.mesh_utils import get_vertices_per_edge

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


@torch.no_grad()
class FittingMonitor():
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=30, ftol=1e-9, gtol=1e-9,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        # super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        self.vpe = None

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None, camera=None, img_path=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = False
        prev_loss = None
        # print('\n')
        for n in range(self.maxiters):
            loss = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break
            
            if self.visualize and n % self.summary_steps == 0:
            # if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach()
                body_joints = model_output.joints.detach()
                utils.visualize_fitting(body_joints, vertices, body_model.faces, camera, img_path, save=False)
                cv2.waitKey()
                # self.mv.update_mesh(vertices.squeeze(),
                #                     body_model.faces)

            prev_loss = loss.item()
            # print('stage fitting loss: ', prev_loss)
        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               origin_mesh=None,
                               joints_conf=None,
                               gt_joints3d=None,
                               joints3d_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               use_3d=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)

        # the vposer++ contains wrist
        append_wrists = False
        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if self.vpe is None:
                f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
                self.vpe = get_vertices_per_edge(
                        body_model.v_template.detach().cpu().numpy(), body_model.faces[f_sel])
                self.connection = torch.tensor(self.vpe, dtype=torch.long, device=body_pose.device)

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,body_model=body_model,           origin_mesh=origin_mesh,
                             gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              gt_joints3d=gt_joints3d,
                              joints3d_conf=joints3d_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              vpe=self.connection,
                              **kwargs)

            if backward:
                total_loss.backward(retain_graph=True, create_graph=create_graph)

            # self.steps += 1
            # if self.visualize and self.steps % self.summary_steps == 0:
            #     model_output = body_model(return_verts=True,
            #                               body_pose=body_pose)
            #     vertices = model_output.vertices.detach().cpu().numpy()

            #     self.mv.update_mesh(vertices.squeeze(),
            #                         body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist

class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 use_joints_conf=True,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 use_3d=False,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior
        
        self.use_3d = use_3d

        self.robustifier = GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def get_bounding_boxes(self, vertices):
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    def compute_edges(self, points, connections):
        edge_points = torch.index_select(
            points, 1, connections.view(-1,)).reshape(points.shape[0], -1, 2, 3)
        return edge_points[:, :, 1] - edge_points[:, :, 0]

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                gt_joints3d=None, joints3d_conf=None,origin_mesh=None, vpe=None,
                **kwargs):

        # edge loss
        gt_edges = self.compute_edges(
            body_model_output.vertices, connections=vpe)
        est_edges = self.compute_edges(
            origin_mesh[None,:], connections=vpe)

        raw_edge_diff = (gt_edges - est_edges)
        edge_diff = raw_edge_diff.pow(2)
        edge_diff = edge_diff.sum()

        # 3d loss
        joints3d_loss = 0.
        diff3d = self.robustifier(body_model_output.vertices[0] - origin_mesh)
        joints3d_loss = (torch.sum(diff3d) *
                    self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
            if float(pprior_loss) > 5e4:
                pprior_loss = 0.
            pprior_loss += body_model_output.body_pose.pow(2).sum() * (self.body_pose_weight * 4) ** 2

        shape_loss = 0.
        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]

        angle_prior_loss = 0.

        pen_loss = 0.0

        total_loss = (joints3d_loss + edge_diff + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss)
        return total_loss