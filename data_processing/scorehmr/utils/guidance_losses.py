import torch

from .geometry import perspective_projection
import numpy as np

def gmof(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure error function.
    Args:
        x : Raw error signal
        sigma : Robustness hyperparameter
    Returns:
        torch.Tensor: Robust error signal
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def shape_loss(betas, init_betas):
    loss = torch.linalg.norm(betas - init_betas, dim=1)
    loss = loss.mean()

    return loss


def contact_loss(pred_verts, camera_translation, batch, frame_num, agent_num):
    total_loss = torch.tensor(0).to("cuda").to(torch.float32).requires_grad_()
    pred_verts = pred_verts + camera_translation.view(-1, 1, 3)

    contact_verts = batch['contact_verts']
    frame_idx = batch['contact_idx']
    verts1_idx = batch['verts1_idx']
    verts2_idx = batch['verts2_idx']

    n_point = verts1_idx[0].shape[0]

    assert len(contact_verts) == frame_num
    pred_verts = pred_verts.reshape(agent_num, frame_num, -1, 3)

    verts1 = pred_verts[0][frame_idx].reshape(-1, 3)
    idx1 = np.array(verts1_idx).reshape(-1,).tolist()

    verts2 = pred_verts[1][frame_idx].reshape(-1, 3)
    idx2 = np.array(verts2_idx).reshape(-1,).tolist()

    verts1 = verts1[idx1].reshape(len(frame_idx), n_point, 3)
    verts2 = verts2[idx2].reshape(len(frame_idx), n_point, 3)

    dist = torch.vmap(lambda x, y: torch.cdist(x, y, p=2) ** 2)(verts1, verts2)

    dist = torch.min(dist.reshape(-1, n_point*n_point), dim=1)[0]

    total_loss = dist.sum()

    return total_loss

def keypoint_fitting_loss(
    model_joints: torch.Tensor,
    camera_translation: torch.Tensor,
    joints_2d: torch.Tensor,
    joints_conf: torch.Tensor,
    camera_center: torch.Tensor,
    focal_length: torch.Tensor,
    img_size: torch.Tensor,
    vis_mask: torch.Tensor,
    sigma: float = 100.,
    step: int = 0,
    time: int = 0,
    cam: int = 0,
) -> torch.Tensor:
    """
    Loss function for model fitting on 2D keypoints.
    Args:
        model_joints       (torch.Tensor) : Tensor of shape [B, NJ, 3] containing the SMPL 3D joint locations.
        camera_translation (torch.Tensor) : Tensor of shape [B, 3] containing the camera translation.
        joints_2d          (torch.Tensor) : Tensor of shape [B, N, 2] containing the target 2D joint locations.
        joints_conf        (torch.Tensor) : Tensor of shape [B, N, 1] containing the target 2D joint confidences.
        camera_center      (torch.Tensor) : Tensor of shape [B, 2] containing the camera center in pixels.
        focal_length       (torch.Tensor) : Tensor of shape [B, 2] containing focal length value in pixels.
        img_size           (torch.Tensor) : Tensor of shape [B, 2] containing the image size in pixels (height, width).
    Returns:
        torch.Tensor: Total loss value.
    """
    img_size = img_size.max(dim=-1)[0]

    # Heuristic for scaling data_weight with resolution used in SMPLify-X
    data_weight = (1000.0 / img_size).reshape(-1, 1, 1).repeat(1, 1, 2)

    # Project 3D model joints
    projected_joints = perspective_projection(
        model_joints, camera_translation, focal_length, camera_center=camera_center
    )
    vis_mask = vis_mask.to(torch.bool)
    not_vis_mask = ~vis_mask
    projected_joints[not_vis_mask] = 0.

    # Compute robust reprojection loss
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (
        (data_weight**2) * (joints_conf**2) * reprojection_error
    ).sum(dim=(1, 2))

    return reprojection_loss


def multiview_loss(
    body_pose_6d: torch.Tensor, consistency_weight: float = 300.0
) -> torch.Tensor:
    """
    Loss function for multiple view refinement.
    Args:
        body_pose_6d : Tensor of shape (V, 23, 6) containing the 6D pose of V views of a person.
        consistency_weight : Pose consistency loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    mean_pose = body_pose_6d.mean(dim=0).unsqueeze(dim=0)
    pose_diff = ((body_pose_6d - mean_pose) ** 2).sum(dim=-1)
    consistency_loss = consistency_weight**2 * pose_diff.sum()
    total_loss = consistency_loss
    return total_loss

def transl_smoothness_loss(transls, frame_num, agent_num):
    transls = transls.reshape(agent_num, frame_num, -1)

    loss = torch.tensor(0).to("cuda").to(torch.float32).requires_grad_()
    for transl in transls:
        loss = loss + torch.linalg.norm(transl[:-1] - transl[1:], dim=1).mean()

    return loss

def smoothness_loss(pred_pose_6d, pred_keypoints_3d, frame_num, agent_num):
    """
    Loss function for temporal smoothness.
    Args:
        pred_pose : Tensor of shape [N, 144] containing the 6D pose of N frames in a video.
    Returns:
        torch.Tensor : Total loss value.
    """
    pred_pose_6d = pred_pose_6d.reshape(agent_num, frame_num, -1)
    # pred_keypoints_3d = pred_keypoints_3d + trans[:,None,:]
    pred_keypoints_3d = pred_keypoints_3d.reshape(agent_num, frame_num, -1, 3)

    
    # pred_verts = pred_verts + trans[:,None,:]

    loss = torch.tensor(0).to("cuda").to(torch.float32).requires_grad_()

    # from utils.FileLoaders import write_obj
    # from utils.smpl_torch_batch import SMPLModel
    # smpl = SMPLModel(model_path=R'smpl\smpl\SMPL_NEUTRAL.pkl')

    # for i, verts in enumerate(pred_verts):
    #     write_obj(verts.detach().cpu().numpy().reshape(-1, 3), smpl.faces, 'output/verts%05d.obj' %i )
    
    # write_obj(pred_keypoints_3d[0][0].detach().cpu().numpy().reshape(-1, 3), [], 'output/joints0.obj')
    # write_obj(pred_keypoints_3d[1][0].detach().cpu().numpy().reshape(-1, 3), [], 'output/joints1.obj')

    # write_obj(pred_verts[0].detach().cpu().numpy().reshape(-1, 3), smpl.faces, 'output/verts0.obj')
    # write_obj(pred_verts[1].detach().cpu().numpy().reshape(-1, 3), smpl.faces, 'output/verts1.obj')
    # write_obj(pred_verts[300].detach().cpu().numpy().reshape(-1, 3), smpl.faces, 'output/verts300.obj')

    # for pose in pred_pose_6d:
    #     loss = loss + torch.linalg.norm(pose[:-1] - pose[1:], dim=1).mean()
    for pose in pred_keypoints_3d:
        acceleration = pose[2:] - 2 * pose[1:-1] + pose[:-2]
        loss = loss + torch.mean(torch.norm(acceleration, dim=-1) ** 2)
        # loss = loss + torch.linalg.norm(pose[:-1] - pose[1:], dim=-1).mean()

    return loss
