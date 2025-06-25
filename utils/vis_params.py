

import numpy as np
import matplotlib.pyplot as plt

def plot_smpl_params_over_time(pose, shape, trans, gt_pose, gt_shape, gt_trans, title_prefix="SMPL", save_path=None):
    """
    Plots the time-series trajectory of SMPL parameters.

    Args:
        smpl_params (dict): Dictionary containing SMPL parameters over time.
        title_prefix (str): Title prefix for plots.
        save_path (str or None): If set, save the plots as a PDF file.
    """
    num_frames = pose.shape[0]
    time = np.arange(num_frames)

    fig, axes = plt.subplots(6, 1, figsize=(15, 18))
    fig.suptitle(f"{title_prefix} Parameters Over Time", fontsize=18)

    # Plot global orientation
    axes[0].set_title("Global Orientation (axis-angle)")
    for i in range(3):
        axes[0].plot(time, pose[:, 3+i], label=f'pose [{i}]')
    axes[0].legend()

    # Plot global orientation
    axes[0].set_title("Global Orientation (axis-angle)")
    for i in range(3):
        axes[0].plot(time, gt_pose[:, 3+i], label=f'gt pose [{i}]')
    axes[0].legend()

    # # Plot body pose (first 9 joints, i.e., 27 dims)
    # axes[1].set_title("Body Pose (axis-angle, first 9 joints)")
    # for i in range(27):
    #     axes[1].plot(time, smpl_params['body_pose'][:, i], label=f'pose[{i}]', alpha=0.6)
    # axes[1].legend(ncol=3, fontsize=8)

    # # Plot body pose norm (total motion magnitude per frame)
    # axes[2].set_title("Body Pose L2 Norm per Frame")
    # pose_norm = np.linalg.norm(smpl_params['body_pose'], axis=1)
    # axes[2].plot(time, pose_norm, label='||body_pose||', color='blue')
    # axes[2].legend()

    # # Plot transl (if available)
    # if 'transl' in smpl_params:
    #     axes[3].set_title("Translation (x, y, z)")
    #     for i in range(3):
    #         axes[3].plot(time, smpl_params['transl'][:, i], label=f'transl[{i}]')
    #     axes[3].legend()
    # else:
    #     axes[3].set_visible(False)

    # # Plot betas (if available)
    # if 'betas' in smpl_params:
    #     axes[4].set_title("Shape Parameters (betas)")
    #     for i in range(smpl_params['betas'].shape[1]):
    #         axes[4].plot(time, smpl_params['betas'][:, i], label=f'beta[{i}]')
    #     axes[4].legend()
    # else:
    #     axes[4].set_visible(False)

    # # Plot angular velocity (optional, finite diff of global_orient)
    # global_vel = np.diff(smpl_params['global_orient'], axis=0)
    # global_vel = np.vstack([global_vel, global_vel[-1]])  # pad for same length
    # axes[5].set_title("Global Orientation Velocity")
    # for i in range(3):
    #     axes[5].plot(time, global_vel[:, i], label=f'd(global_orient)[{i}]')
    # axes[5].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
