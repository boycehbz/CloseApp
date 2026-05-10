import os
import numpy as np
import torch
import math
import cv2

from cliff.cliff.utils.geometry import batch_rodrigues
from scorehmr.configs import model_config
from scorehmr.models.model_utils import load_pare, load_diffusion_model
from scorehmr.utils import StandarizeImageFeatures, prepare_smpl_params
from utils.FileLoaders import write_obj, save_pkl
from utils.module_utils import save_camparam, vis_img, draw_keyp
# from utils.renderer_pyrd import Renderer
from utils.renderer_moderngl import Renderer
from utils.smpl_torch_batch import SMPLModel
from tqdm import tqdm
from utils.rotation_conversions import matrix_to_axis_angle

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
NUM_SAMPLES = 1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

class ScoreHMR:
    def __init__(self, device):
        self.smpl = SMPLModel(model_path='smpl/smpl/SMPL_NEUTRAL.pkl')
        self.model_cfg = model_config()
        self.device = device
        self.pare = load_pare(self.model_cfg.SMPL).to(self.device)
        self.pare.eval()
        self.img_feat_standarizer = StandarizeImageFeatures(
            backbone=self.model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
            use_betas=False,
            device=self.device,
        )
        self.extra_args = {
            "keypoint_guidance": True,
            "temporal_guidance": True,
            "use_default_ckpt": True,
            "device": self.device,
        }
        self.diffusion_model = load_diffusion_model(self.model_cfg, **self.extra_args)
        self.faces = self.smpl.faces

    def iterate(self, mv_batch, track_num, frame_num, image_folder, viz=False, save_results=False):
        output_folder = os.path.join("output/scorehmr", os.path.basename(image_folder)) 
        os.makedirs(output_folder, exist_ok=True)

        batch = mv_batch[0]
        batch_size = batch["joints_2d"].size(0)
        agent_num = batch_size // frame_num

        # Get PARE image features.
        pare_out = {'pose_feats':[],'cam_shape_feats':[],}
        with torch.no_grad():
            sub_batchsize = 20
            if len(batch["img"]) > sub_batchsize:
                n_sub_batch = math.ceil(len(batch["img"]) / sub_batchsize)
                for n in range(n_sub_batch):
                    sub_batch = self.pare(batch["img"][n*sub_batchsize:(n+1)*sub_batchsize], get_feats=True)
                    pare_out['pose_feats'].append(sub_batch['pose_feats'])
                    pare_out['cam_shape_feats'].append(sub_batch['cam_shape_feats'])

                pare_out['pose_feats'] = torch.cat(pare_out['pose_feats'])
                pare_out['cam_shape_feats'] = torch.cat(pare_out['cam_shape_feats'])
            else:
                pare_out = self.pare(batch["img"], get_feats=True)
            torch.cuda.empty_cache()

        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = self.img_feat_standarizer(cond_feats)  # normalize image features

        # Iterative refinement with ScoreHMR.
        # print(f'=> Running ScoreHMR for tracklet {track_idx+1}/{len(data)}')
        with torch.no_grad():
            dm_out = self.diffusion_model.sample(
                mv_batch, cond_feats, frame_num, batch_size=batch_size * NUM_SAMPLES
            )
        pred_smpl_params, trans = prepare_smpl_params(
            dm_out['x_0'],
            dm_out["camera_translation"],
            num_samples = NUM_SAMPLES,
            use_betas = False,
            pred_betas=dm_out["betas"],
            frame_num=frame_num,
            agent_num=agent_num,
        )
        smpl_out = self.diffusion_model.smpl(**pred_smpl_params, pose2rot=False)

        # pose
        body_pose_mat = smpl_out.body_pose
        global_orient_mat = smpl_out.global_orient
        pose_mat = torch.cat((global_orient_mat, body_pose_mat), dim=1)
        pose_vec = matrix_to_axis_angle(pose_mat).cpu().numpy().reshape(track_num, frame_num, 72)

        # betas
        betas = smpl_out.betas.cpu().numpy().reshape(track_num, frame_num, 10)
        params = {
            "pose": pose_vec,
            "trans": trans,
            "betas": betas,
        }

        vertices = smpl_out.vertices.cpu().numpy()
        vertices = vertices + trans.reshape(-1, 1, 3)
        vertices = vertices.reshape(track_num, frame_num, -1, 3)
        if viz or save_results:
            print('Rendering')
            joints_2d = batch['joints_2d'].reshape(agent_num, frame_num, -1, 2).detach().cpu().numpy()
            for frame_idx, img in tqdm(enumerate(sorted(os.listdir(image_folder))), total=len(os.listdir(image_folder))):
                verts = vertices[:, frame_idx, :, :]
                imgname = os.path.join(image_folder, img)
                img_origin = cv2.imread(imgname)
                focal_length = batch["focal_length"].cpu().numpy()[0, 0]
                front_view = self.vis_results(img_origin, focal_length, verts)

                for joint in joints_2d:
                    front_view = draw_keyp(front_view, joint[frame_idx], format='coco17')

                if save_results:
                    cv2.imwrite(os.path.join(output_folder, img), front_view)

                # rendered.append(front_view)
                if viz:
                    vis_img('img', front_view)
                # output_dir = f"output/front_view"
                # os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
                # cv2.imwrite(output_path, front_view)

                # output_dir = f"output/side_view"
                # os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
                # cv2.imwrite(output_path, side_view)

        return params, vertices

    def vis_results(self, img_origin, focal_length, verts, viz=False):
        img = img_origin.copy()
        renderer = Renderer(focal_length=focal_length, img_w=img.shape[1], img_h=img.shape[0],
                            faces=self.smpl.faces,
                            same_mesh_color=True)
        front_view = renderer.render_front_view(verts, bg_img_rgb=img.copy())
        side_view = renderer.render_side_view(verts)
        top_view = renderer.render_top_view(verts)

        rendered = np.concatenate((img_origin, front_view, side_view, top_view), axis=1)
        renderer.delete()

        return rendered

    def save_resutls(self, image_folder, params, img, vertices, focal_length):
        output_folder = os.path.join("output/scorehmr", os.path.basename(image_folder)) 
        os.makedirs(output_folder, exist_ok=True)

        name = sorted([img_name for img_name in os.listdir(image_folder)])

        print('Saving results')
        for idx, n in tqdm(enumerate(name), total=len(name)):
            # cv2.imwrite(os.path.join(output_folder, n), rendered[idx])

            mesh_folder = os.path.join(output_folder, "meshes/" + n[:-4])
            os.makedirs(mesh_folder, exist_ok=True)
            verts = vertices[:, idx, :, :]
            for i, v in enumerate(verts):
                write_obj(v, self.faces, os.path.join(mesh_folder, '%04d.obj' %i))

            params_folder = os.path.join(output_folder, "params/" + n[:-4])
            os.makedirs(params_folder, exist_ok=True)
            params_frame = {
                "pose": params["pose"][:, idx, :],
                "trans": params["trans"][:, idx, :],
                "betas": params["betas"][:, idx, :],
            }
            save_pkl(os.path.join(params_folder, '0000.pkl'), params_frame)

            camparams_folder = os.path.join(output_folder, "camparams/" + n[:-4])
            os.makedirs(camparams_folder, exist_ok=True)
            intri = np.eye(3)
            extri = np.eye(4)
            intri[0][0] = focal_length.cpu().item()
            intri[1][1] = focal_length.cpu().item()
            intri[0][2] = img.shape[1] / 2
            intri[1][2] = img.shape[0] / 2
            save_camparam(os.path.join(camparams_folder, "camparams.txt"), [intri], [extri])