import torch
import numpy as np
import torch
import os
import numpy as np
import torch.nn as nn
from submodules import smplx
import trimesh
from scene.dataset_mono import MonoDataset_train, MonoDataset_test, MonoDataset_novel_pose, MonoDataset_novel_view
from utils.general_utils import worker_init_fn
from utils.system_utils import mkdir_p
from model.network import POP_no_unet
from utils.general_utils import load_masks
from gaussian_renderer import render_batch
from os.path import join
import torch.nn as nn
from model.modules  import UnetNoCond5DS
from utils.smpl_torch_batch import SMPLModel
# from utils.renderer_pyrd import Renderer
from utils.renderer_moderngl import Renderer
from utils.module_utils import vis_img, OneEuroFilter, save_camparam, overlay_mask_on_image
from utils.eval_utils import HumanEval
from utils.FileLoaders import *
from utils.rotation_conversions import *
import cv2
from utils.logger import Logger
import time
from sdf import SDFLoss
from utils.mesh_intersection.bvh_search_tree import BVH
import utils.mesh_intersection.loss as collisions_loss
from utils.mesh_intersection.filter_faces import FilterFaces
from model.interhuman_diffusion_phys import interhuman_diffusion_phys
from utils.geometry import perspective_projection
from scipy import signal
from utils.vis_params import plot_smpl_params_over_time
from tqdm import tqdm
from utils.module_utils import draw_keyp

class AvatarModel:
    def __init__(self, model_parms, net_parms, opt_parms, load_iteration=None, train=True):

        self.smpl_neutral = SMPLModel(device=torch.device('cpu'), model_path='data/smpl/smpl/SMPL_NEUTRAL.pkl')
        self.smpl_male = SMPLModel(device=torch.device('cpu'), model_path='data/smpl/smpl/SMPL_MALE.pkl')
        self.smpl_female = SMPLModel(device=torch.device('cpu'), model_path='data/smpl/smpl/SMPL_FEMALE.pkl')
        
        self.smpl_gpu = SMPLModel(device=torch.device('cuda'), model_path='data/smpl/smpl/SMPL_NEUTRAL.pkl')
        
        self.sdf_loss = SDFLoss(self.smpl_gpu.faces, self.smpl_gpu.faces, robustifier=None).cuda()


        self.loss_weights = {'reproj': 0.5, 
                             'motion_prior': 1.0, 
                             'transl_prior': 0.0, 
                             'shape_prior': 0.0, 
                             'smoothness': 0.1, 
                             'pen_loss': 0.000,
                             'contact_loss':0.0, }

        self.loss = {'reproj': 0.0, 
                    'motion_prior': 0.0, 
                    'transl_prior': 0.0, 
                    'shape_prior': 0.0, 
                    'smoothness': 0.0, 
                    'pen_loss': 0.0,
                    'contact_loss':0.0, }

        self.pen_threshold = 500.
        self.search_tree = BVH(max_collisions=8)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
                                                         point2plane=False,
                                                         vectorized=True)

        # Create the log for the current experiment
        mon, day, hour, min, sec = time.localtime(time.time())[1:6]
        self.logger = Logger(os.path.join(model_parms.model_path, 'log.txt'), title="template")
        self.logger.set_names([model_parms.model_path.split('/')[-1]])
        self.logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
        self.logger.set_names(['Total', 'Motion', 'MPVPE', 'MPJPE', 'PA-MPJPE', 'Accel', 'A_PD'])

        self.best_MPJPE = 99999999.
        self.best_opt_loss = 99999999.
        self.best_MPJPE_count = 0
        self.best_opt_loss_count = 0

        self.best_MPJPE_state = None
        self.best_MPJPE_path = None
        self.best_opt_loss_state = None
        self.best_opt_loss_path = None

        self.model_parms = model_parms
        self.net_parms = net_parms
        self.opt_parms = opt_parms
        self.model_path = model_parms.model_path
        self.loaded_iter = None
        self.train = train
        self.train_mode = model_parms.train_mode
        self.gender = self.model_parms.smpl_gender

        self.num_agent = 2


        if train:
            self.batch_size = self.model_parms.batch_size
        else:
            self.batch_size = 1

        if train:
            split = 'train'
        else:
            split = 'test'

        self.train_dataset  = MonoDataset_train(model_parms)
        self.frame_length = self.train_dataset.frame_length
        self.smpl_data = self.train_dataset.smpl_data

        # partial code derive from POP (https://github.com/qianlim/POP)
        assert model_parms.smpl_type in ['smplx', 'smpl']
        if model_parms.smpl_type == 'smplx':
            self.smpl_model = smplx.SMPLX(model_path=self.model_parms.smplx_model_path, gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = self.batch_size*self.num_agent*self.frame_length).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smplx')
            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smplx_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path, split, 'smplx_cano_joint_mat.pth')
            joint_num = 55
        
        else:
            self.smpl_model = smplx.SMPL(model_path=self.model_parms.smpl_model_path, gender = self.gender, batch_size = self.batch_size*self.num_agent*self.frame_length).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smpl')

            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'data', 'lbs_map_smpl_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path,  split, 'smpl_cano_joint_mat.pth')
            joint_num = 24

        self.uv_coord_map = uv_coord_map
        self.valid_idx = valid_idx

        if model_parms.fixed_inp:
            fix_inp_map = torch.from_numpy(np.load(inp_map_path)['posmap' + str(self.model_parms.inp_posmap_size)].transpose(2,0,1)).cuda()
            self.fix_inp_map = fix_inp_map[None].expand(self.batch_size, -1, -1, -1)
        
        ## query_map store the sampled points from the cannonical smpl mesh, shape as [512. 512, 3] 
        query_map = torch.from_numpy(np.load(query_map_path)['posmap' + str(self.model_parms.query_posmap_size)]).reshape(-1,3)
        query_points = query_map.cuda()[valid_idx, :].contiguous()
        self.query_points = query_points[None].expand(self.batch_size*self.num_agent*self.frame_length, -1, -1)
        
        # we fix the opacity and rots of 3d gs as described in paper 
        self.fix_opacity = torch.ones((self.query_points.shape[1]*self.num_agent*self.frame_length, 1)).cuda()
        rots = torch.zeros((self.query_points.shape[1]*self.num_agent*self.frame_length, 4), device="cuda")
        rots[:, 0] = 1
        self.fix_rotation = rots
        
        # we save the skinning weights from the cannonical mesh
        query_lbs = torch.from_numpy(np.load(query_lbs_path)).reshape(self.model_parms.query_posmap_size*self.model_parms.query_posmap_size, joint_num)
        self.query_lbs = query_lbs.cuda()[valid_idx, :][None].expand(self.batch_size*self.num_agent*self.frame_length, -1, -1).contiguous()
        
        self.inv_mats = torch.linalg.inv(torch.load(mat_path)).expand(self.batch_size*self.num_agent*self.frame_length, -1, -1, -1).cuda()
        print('inv_mat shape: ', self.inv_mats.shape)

        num_training_frames = self.train_dataset.data_length
        self.num_training_frames = num_training_frames
        param = []

        self.prior_weight = torch.ones((self.num_training_frames, self.num_agent), dtype=torch.float32).cuda()  # self.calc_weight_from_velocity(self.train_dataset)
        # self.prior_weight[20:28,0] = 0.
        # self.prior_weight[40:77,0] = 0.

        # if not torch.is_tensor(self.smpl_data['pred_shape']):
        #     self.betas = torch.from_numpy(self.smpl_data['pred_shape'][0])[None].expand(self.batch_size, -1).cuda()
        # else:
        #     self.betas = self.smpl_data['pred_shape'][0,0][None].expand(self.batch_size, -1).cuda()

        if model_parms.smpl_type == 'smplx':
            self.pose = torch.nn.Embedding(num_training_frames, 66, _weight=self.train_dataset.pose_data, sparse=True).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
            param += list(self.transl.parameters())
        else:
            self.pose_0 = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data[:,0], sparse=True).cuda()
            param += list(self.pose_0.parameters())
            self.pose_1 = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data[:,1], sparse=True).cuda()
            param += list(self.pose_1.parameters())

            self.transl_0 = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data[:,0], sparse=True).cuda()
            param += list(self.transl_0.parameters())
            self.transl_1 = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data[:,1], sparse=True).cuda()
            param += list(self.transl_1.parameters())

            self.betas_0 = torch.nn.Embedding(1, 10, _weight=self.train_dataset.shape_data[0][None], sparse=True).cuda()
            param += list(self.betas_0.parameters())
            self.betas_1 = torch.nn.Embedding(1, 10, _weight=self.train_dataset.shape_data[1][None], sparse=True).cuda()
            param += list(self.betas_1.parameters())

            self.init_pose = self.train_dataset.pose_data.cuda()
            self.init_transl = self.train_dataset.transl_data.cuda()
            self.init_shapes = self.train_dataset.shape_data.cuda()

            # self.init_transl[:,0,2] += 1
            # self.init_transl[:,1,2] -= 1
            # self.init_shapes[0][0] = 5
            # self.init_shapes[1][0] = -5

        self.optimizer_pose = torch.optim.SparseAdam(param, 5.0e-3)
        
        bg_color = [1, 1, 1] if model_parms.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.rotation_activation = torch.nn.functional.normalize
        self.sigmoid_activation = nn.Sigmoid()
        
        self.net_set(self.model_parms.train_stage)

    def calc_weight_from_velocity(self, train_dataset):
        num_frame, num_agent = train_dataset.pose_data.shape[:2]

        pose = train_dataset.pose_data.reshape(-1, 72)
        shape = train_dataset.shape_data[None,:].expand(num_frame, -1, -1).reshape(-1, 10)
        trans = train_dataset.transl_data.reshape(-1, 3)

        verts, joints = self.smpl_neutral(shape, pose, trans)
        joints = joints.reshape(num_frame, num_agent, -1, 3)

        vel = abs(joints[1:] - joints[:-1])
        print(1)

    def save_rendered(self, images, gt_images, epoch, names, texture_map):
        images = (images.detach().cpu().numpy() * 255).transpose((0,2,3,1)).astype(np.uint8)
        gt_images = (gt_images.detach().cpu().numpy() * 255).transpose((0,2,3,1)).astype(np.uint8)

        folder = os.path.join(self.model_path, 'log', 'epoch%05d' %epoch)
        os.makedirs(folder, exist_ok=True)

        texture_map = texture_map.cpu().numpy().transpose((0,2,1)).reshape(2, self.model_parms.query_posmap_size, self.model_parms.query_posmap_size, 3)

        mask = self.valid_idx.reshape(self.model_parms.query_posmap_size, self.model_parms.query_posmap_size, 1).detach().cpu().numpy().astype(np.float32)

        cv2.imwrite(os.path.join(self.model_path, 'log', 'epoch%05d_c%02d_f%s.jpg' %(epoch, 0, names[0][0])), texture_map[0]*255*mask[:,:,::-1])
        cv2.imwrite(os.path.join(self.model_path, 'log', 'epoch%05d_c%02d_f%s.jpg' %(epoch, 1, names[0][0])), texture_map[1]*255*mask[:,:,::-1])

        for img, gt_img, name in zip(images, gt_images, names):
            img = np.concatenate((img[:,:,::-1], gt_img[:,:,::-1]), axis=1)
            img_path = os.path.join(folder, name[0] + '.jpg')

            cv2.imwrite(img_path, img)

        

    def eval_smpl(self, total_loss, motion_loss):
        if self.train_dataset.gt_pose is None:
            print(' Total: %.5f, Motion: %.5f' %(total_loss, motion_loss))
            return 9999999.
        eval_tool = HumanEval(name='Hi4D', smpl=self.smpl_neutral)

        gt_verts_total, opt_verts_total = [], []
        vertex_errors, errors, error_pas, accels = [], [], [], []
        for agent in range(self.num_agent):

            gt_pose = self.train_dataset.gt_pose[:,agent].contiguous()
            gt_trans = self.train_dataset.gt_trans[:,agent].contiguous()
            gt_shape = self.train_dataset.gt_shape[0,agent][None,:].expand(gt_pose.shape[0], -1)

            if self.train_dataset.gt_gender[agent] == 'male':
                smpl_model = self.smpl_male
            elif self.train_dataset.gt_gender[agent] == 'female':
                smpl_model = self.smpl_female
            else:
                smpl_model = self.smpl_neutral

            gt_verts, _ = smpl_model(gt_shape, gt_pose, gt_trans)

            if agent == 0:
                opt_pose = self.pose_0.weight.detach().cpu()
                opt_trans = self.transl_0.weight.detach().cpu()
                opt_shape = self.betas_0.weight.detach().expand(opt_pose.shape[0], -1).cpu()
            else:
                opt_pose = self.pose_1.weight.detach().cpu()
                opt_trans = self.transl_1.weight.detach().cpu()
                opt_shape = self.betas_1.weight.detach().expand(opt_pose.shape[0], -1).cpu()

            opt_verts, _ = self.smpl_neutral(opt_shape, opt_pose, opt_trans)

            gt_verts_total.append(gt_verts)
            opt_verts_total.append(opt_verts)

            vertex_error, error, error_pa, accel = eval_tool.eval_verts(opt_verts.detach().cpu().numpy(), gt_verts.detach().cpu().numpy())
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            accels += accel

        ### penetration
        A_verts = opt_verts_total[0].unsqueeze(dim=1)
        B_verts = opt_verts_total[1].unsqueeze(dim=1)
        hand_verts = torch.cat([A_verts, B_verts], dim=1).cuda()
        _, _, collision_loss_origin_scale = self.sdf_loss(hand_verts, return_per_vert_loss=True, return_origin_scale_loss=True)

        # There is a bug in this loss (produce different loss for the same verts)
        A_PD = np.array((1000* collision_loss_origin_scale.detach().cpu().numpy().mean(axis=1)).tolist()).mean()

        vertex_error = np.array(vertex_errors).mean()
        error = np.array(errors).mean()
        error_pa = np.array(error_pas).mean()
        accel = np.array(accels).mean()
        print(' Total: %.5f, Motion: %.5f, MPVPE: %.5f, MPJPE: %.5f PA-MPJPE: %.5f, Accel: %.5f, A_PD: %.5f' %(total_loss, motion_loss, vertex_error, error, error_pa, accel, A_PD))
        self.logger.append([total_loss, motion_loss, vertex_error, error, error_pa, accel, A_PD])
        return error

    def vis_smpl_frame(self, save_render=False, viz=False, save_mesh=False, save_params=False, epoch=0, smooth=False):
        pose_0 = self.pose_0.weight.detach().cpu()
        trans_0 = self.transl_0.weight.detach().cpu()
        shape_0 = self.betas_0.weight.detach().expand(pose_0.shape[0], -1).cpu()
        pose_1 = self.pose_1.weight.detach().cpu()
        trans_1 = self.transl_1.weight.detach().cpu()
        shape_1 = self.betas_1.weight.detach().expand(pose_1.shape[0], -1).cpu()

        m_type = 'normal'

        pose = torch.cat([pose_0, pose_1], dim=0)
        trans = torch.cat([trans_0, trans_1], dim=0)
        shape = torch.cat([shape_0, shape_1], dim=0)

        img_folder = os.path.join(self.train_dataset.data_folder, 'images')
        files = sorted(os.listdir(img_folder))

        verts, joints = self.smpl_neutral(shape, pose, trans)
        verts_0, verts_1 = verts[:self.num_training_frames], verts[self.num_training_frames:]

        if self.train_dataset.gt_pose is not None:
            gt_verts, gt_poses, gt_shapes, gt_transs = [], [], [], []
            for agent in range(self.num_agent):

                gt_pose = self.train_dataset.gt_pose[:,agent].contiguous()
                gt_trans = self.train_dataset.gt_trans[:,agent].contiguous()
                gt_shape = self.train_dataset.gt_shape[0,agent][None,:].expand(gt_pose.shape[0], -1)

                if self.train_dataset.gt_gender[agent] == 'male':
                    smpl_model = self.smpl_male
                elif self.train_dataset.gt_gender[agent] == 'female':
                    smpl_model = self.smpl_female
                else:
                    smpl_model = self.smpl_neutral

                gt_vert, _ = smpl_model(gt_shape, gt_pose, gt_trans)
                gt_verts.append(gt_vert)
                gt_poses.append(gt_pose)
                gt_shapes.append(gt_shape)
                gt_transs.append(gt_trans)

            gt_verts_0, gt_verts_1 = gt_verts[0], gt_verts[1]
            gt_pose_0, gt_pose_1 = gt_poses[0], gt_poses[1]
            gt_shape_0, gt_shape_1 = gt_shapes[0], gt_shapes[1]
            gt_trans_0, gt_trans_1 = gt_transs[0], gt_transs[1]
        else:
            gt_verts_0, gt_verts_1 = verts_0, verts_1

        for i, (name, vert_0, vert_1, gt_vert_0, gt_vert_1) in enumerate(zip(files, verts_0, verts_1, gt_verts_0, gt_verts_1)):
            img = cv2.imread(os.path.join(img_folder, name))
            idx = name.split('.')[0]

            vert = torch.cat([vert_0[None,:], vert_1[None,:]], dim=0)
            gt_vert = torch.cat([gt_vert_0[None,:], gt_vert_1[None,:]], dim=0)

            renderer = Renderer(focal_length=self.train_dataset.intrinsic[0][0], center=(self.train_dataset.intrinsic[0][2], self.train_dataset.intrinsic[1][2]), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_neutral.faces, same_mesh_color=True)
            pred_smpl = renderer.render_front_view(vert.detach().numpy(), bg_img_rgb=img.copy())
            pred_side = renderer.render_side_view(vert.detach().numpy())
            renderer.delete()

            renderer = Renderer(focal_length=self.train_dataset.gt_intrinsic[0][0], center=(self.train_dataset.gt_intrinsic[0][2], self.train_dataset.gt_intrinsic[1][2]), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_neutral.faces, same_mesh_color=True)
            gt_smpl = renderer.render_front_view(gt_vert.detach().numpy(), bg_img_rgb=img.copy())
            gt_side = renderer.render_side_view(gt_vert.detach().numpy())
            renderer.delete()
            background = np.ones_like(pred_smpl)*255

            smpl = np.concatenate((img, pred_smpl, gt_smpl), axis=1)
            side = np.concatenate((background, pred_side, gt_side), axis=1)

            pred_smpl = np.concatenate((smpl, side), axis=0)

            if viz:
                vis_img('img', pred_smpl)
            if save_render:
                save_path = os.path.join(self.model_path, 'vis_frame', 'images', idx)
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(os.path.join(save_path, 'epoch%04d_%s' %(epoch, name)), pred_smpl)
            if save_mesh:
                save_path = os.path.join(self.model_path, 'vis_frame', 'meshes', idx)
                os.makedirs(save_path, exist_ok=True)
                
                for agent in range(self.num_agent):
                    write_obj(gt_vert[agent], self.smpl_neutral.faces, os.path.join(save_path, 'epoch%04d_%d_gt.obj' %(epoch, agent)))
                    write_obj(vert[agent], self.smpl_neutral.faces, os.path.join(save_path, 'epoch%04d_%d_pred.obj' %(epoch, agent)))

            if save_params:
                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'params', idx)
                os.makedirs(save_path, exist_ok=True)
                pose_param = torch.cat([pose_0[i][None,:], pose_1[i][None,:]], dim=0)
                shape_param = torch.cat([shape_0[i][None,:], shape_1[i][None,:]], dim=0)
                trans_param = torch.cat([trans_0[i][None,:], trans_1[i][None,:]], dim=0)

                gt_pose_param = torch.cat([gt_pose_0[i][None,:], gt_pose_1[i][None,:]], dim=0)
                gt_shape_param = torch.cat([gt_shape_0[i][None,:], gt_shape_1[i][None,:]], dim=0)
                gt_trans_param = torch.cat([gt_trans_0[i][None,:], gt_trans_1[i][None,:]], dim=0)

                data = {'pose':pose_param.detach().cpu().numpy(),
                        'trans':trans_param.detach().cpu().numpy(),
                        'betas':shape_param.detach().cpu().numpy(),
                        'gt_pose':gt_pose_param.detach().cpu().numpy(),
                        'gt_trans':gt_trans_param.detach().cpu().numpy(),
                        'gt_betas':gt_shape_param.detach().cpu().numpy(),}

                save_pkl(os.path.join(save_path, '0000.pkl'), data)

                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'camparams', idx)
                os.makedirs(save_path, exist_ok=True)

                intri = self.train_dataset.intrinsic
                extri = self.train_dataset.extrinsic
                save_camparam(os.path.join(save_path, 'camparams.txt'), [intri], [extri])

    def vis_smpl(self, save_render=False, viz=False, save_mesh=False, save_params=False, epoch=0, smooth=False):
        pose_0 = self.pose_0.weight.detach().cpu()
        trans_0 = self.transl_0.weight.detach().cpu()
        shape_0 = self.betas_0.weight.detach().expand(pose_0.shape[0], -1).cpu()
        pose_1 = self.pose_1.weight.detach().cpu()
        trans_1 = self.transl_1.weight.detach().cpu()
        shape_1 = self.betas_1.weight.detach().expand(pose_1.shape[0], -1).cpu()

        if smooth:
            one_euro = False
            if one_euro:
                t = np.arange(pose_0.shape[0])
                min_cutoff = 0.004 #0.004
                beta = 10.0

                rotation_matrix = axis_angle_to_matrix(pose_0.reshape(-1, 3))
                rotation_6d = matrix_to_rotation_6d(rotation_matrix)
                rotation_6d = rotation_6d.reshape(pose_0.shape[0], -1).detach().numpy()

                filterdata = np.zeros_like(rotation_6d)
                filterdata[0] = rotation_6d[0]
                one_euro_filter = OneEuroFilter(
                    t[0], rotation_6d[0],
                    min_cutoff=min_cutoff,
                    beta=beta
                )
                for i in range(1, len(t)):
                    filterdata[i] = one_euro_filter(t[i], rotation_6d[i])

                filterdata = torch.from_numpy(filterdata.reshape(-1, 6)).float()
                filterdata = rotation_6d_to_matrix(filterdata)
                filterdata = matrix_to_axis_angle(filterdata)
                pose_0 = filterdata.reshape(pose_0.shape[0], -1)

                rotation_matrix = axis_angle_to_matrix(pose_1.reshape(-1, 3))
                rotation_6d = matrix_to_rotation_6d(rotation_matrix)
                rotation_6d = rotation_6d.reshape(pose_1.shape[0], -1).detach().numpy()

                filterdata = np.zeros_like(rotation_6d)
                filterdata[0] = rotation_6d[0]
                one_euro_filter = OneEuroFilter(
                    t[0], rotation_6d[0],
                    min_cutoff=min_cutoff,
                    beta=beta
                )
                for i in range(1, len(t)):
                    filterdata[i] = one_euro_filter(t[i], rotation_6d[i])

                filterdata = torch.from_numpy(filterdata.reshape(-1, 6)).float()
                filterdata = rotation_6d_to_matrix(filterdata)
                filterdata = matrix_to_axis_angle(filterdata)
                pose_1 = filterdata.reshape(pose_1.shape[0], -1)

            else:
                pose_b, pose_a = signal.butter(2, 0.1, 'lowpass')
                trans_b, trans_a = signal.butter(1, 0.01, 'lowpass')

                rotation_matrix = axis_angle_to_matrix(pose_0.reshape(-1, 3))
                rotation_6d = matrix_to_rotation_6d(rotation_matrix)
                rotation_6d = rotation_6d.reshape(pose_0.shape[0], -1).detach().numpy()

                filterdata = rotation_6d.copy() #[:,pose_ind].copy()
                filterdata = signal.filtfilt(pose_b, pose_a, filterdata.T).T.copy()  # butterworth filter

                filterdata = torch.from_numpy(filterdata.reshape(-1, 6)).float()
                filterdata = rotation_6d_to_matrix(filterdata)
                filterdata = matrix_to_axis_angle(filterdata)
                pose_0 = filterdata.reshape(pose_0.shape[0], -1)

                rotation_matrix = axis_angle_to_matrix(pose_1.reshape(-1, 3))
                rotation_6d = matrix_to_rotation_6d(rotation_matrix)
                rotation_6d = rotation_6d.reshape(pose_1.shape[0], -1).detach().numpy()

                filterdata = rotation_6d.copy() #[:,pose_ind].copy()
                filterdata = signal.filtfilt(pose_b, pose_a, filterdata.T).T.copy()  # butterworth filter

                filterdata = torch.from_numpy(filterdata.reshape(-1, 6)).float()
                filterdata = rotation_6d_to_matrix(filterdata)
                filterdata = matrix_to_axis_angle(filterdata)
                pose_1 = filterdata.reshape(pose_1.shape[0], -1)

                trans_0 = trans_0.detach().cpu().numpy()
                trans_0 = signal.filtfilt(trans_b, trans_a, trans_0.T).T.copy()
                trans_0 = torch.from_numpy(trans_0).float()

                trans_1 = trans_1.detach().cpu().numpy()
                trans_1 = signal.filtfilt(trans_b, trans_a, trans_1.T).T.copy()
                trans_1 = torch.from_numpy(trans_1).float()

            m_type = 'smooth'
        else:
            m_type = 'normal'

        pose = torch.cat([pose_0, pose_1], dim=0)
        trans = torch.cat([trans_0, trans_1], dim=0)
        shape = torch.cat([shape_0, shape_1], dim=0)

        img_folder = os.path.join(self.train_dataset.data_folder, 'images')
        files = sorted(os.listdir(img_folder))

        verts, joints = self.smpl_neutral(shape, pose, trans)
        verts_0, verts_1 = verts[:self.num_training_frames], verts[self.num_training_frames:]
        # shape_0, shape_1 = shape[:self.num_training_frames], shape[self.num_training_frames:]
        # pose_0, pose_1 = pose[:self.num_training_frames], pose[self.num_training_frames:]
        # trans_0, trans_1 = trans[:self.num_training_frames], trans[self.num_training_frames:]


        if self.train_dataset.gt_pose is not None:
            gt_verts, gt_poses, gt_shapes, gt_transs = [], [], [], []
            for agent in range(self.num_agent):

                gt_pose = self.train_dataset.gt_pose[:,agent].contiguous()
                gt_trans = self.train_dataset.gt_trans[:,agent].contiguous()
                gt_shape = self.train_dataset.gt_shape[0,agent][None,:].expand(gt_pose.shape[0], -1)

                if self.train_dataset.gt_gender[agent] == 'male':
                    smpl_model = self.smpl_male
                elif self.train_dataset.gt_gender[agent] == 'female':
                    smpl_model = self.smpl_female
                else:
                    smpl_model = self.smpl_neutral

                gt_vert, _ = smpl_model(gt_shape, gt_pose, gt_trans)
                gt_verts.append(gt_vert)
                gt_poses.append(gt_pose)
                gt_shapes.append(gt_shape)
                gt_transs.append(gt_trans)

            gt_verts_0, gt_verts_1 = gt_verts[0], gt_verts[1]
            gt_pose_0, gt_pose_1 = gt_poses[0], gt_poses[1]
            gt_shape_0, gt_shape_1 = gt_shapes[0], gt_shapes[1]
            gt_trans_0, gt_trans_1 = gt_transs[0], gt_transs[1]
        else:
            gt_verts_0, gt_verts_1 = verts_0, verts_1

        # plot_smpl_params_over_time(pose_0, shape_0, trans_0, gt_pose_0, gt_shape_0, gt_trans_0)

        for i, (name, vert_0, vert_1, gt_vert_0, gt_vert_1) in tqdm(enumerate(zip(files, verts_0, verts_1, gt_verts_0, gt_verts_1)), total=len(files)):
            img = cv2.imread(os.path.join(img_folder, name))
            idx = name.split('.')[0]

            keypoints = self.train_dataset.keypoints[i].detach().cpu().numpy()

            vis_pen = True
            if vis_pen:
                batch_size = 2
                vertices = torch.cat([vert_0[None,:], vert_1[None,:]], dim=0).cuda()
                face_tensor = torch.tensor(self.smpl_gpu.faces.astype(np.int64), dtype=torch.long,
                                        device=vertices.device).unsqueeze_(0).repeat([batch_size,
                                                                                1, 1])
                bs, nv = vertices.shape[:2] # nv: 6890
                bs, nf = face_tensor.shape[:2] # nf: 13776
                faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
                faces_idx = faces_idx.reshape(bs // 2, -1, 3)
                triangles = vertices.view([-1, 3])[faces_idx]

                print_timings = False
                with torch.no_grad():
                    if print_timings:
                        start = time.time()
                    collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
                    if print_timings:
                        torch.cuda.synchronize()
                        print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

                    if False:
                        if print_timings:
                            start = time.time()
                        collision_idxs = self.filter_faces(collision_idxs)
                        if print_timings:
                            torch.cuda.synchronize()
                            print('Collision filtering: {:5f}ms'.format((time.time() -
                                                                        start) * 1000))

                if print_timings:
                        start = time.time()
                pen_loss = self.pen_distance(triangles, collision_idxs)
                if print_timings:
                    torch.cuda.synchronize()
                    print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

                pen_loss = pen_loss.detach().cpu().numpy()

            vert = torch.cat([vert_0[None,:], vert_1[None,:]], dim=0)
            gt_vert = torch.cat([gt_vert_0[None,:], gt_vert_1[None,:]], dim=0)

            renderer = Renderer(focal_length=self.train_dataset.intrinsic[0][0], center=(self.train_dataset.intrinsic[0][2], self.train_dataset.intrinsic[1][2]), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_neutral.faces, same_mesh_color=True)
            pred_smpl = renderer.render_front_view(vert.detach().numpy(), bg_img_rgb=img.copy())
            pred_side = renderer.render_side_view(vert.detach().numpy())
            pred_top = renderer.render_top_view(vert.detach().numpy())


            gt_smpl = renderer.render_front_view(gt_vert.detach().numpy(), bg_img_rgb=img.copy())
            gt_side = renderer.render_side_view(gt_vert.detach().numpy())
            gt_top = renderer.render_top_view(gt_vert.detach().numpy())
            renderer.delete()

            background = np.ones_like(pred_smpl)*255
            if vis_pen:
                background = cv2.putText(background, 'Pen: ' + str(pen_loss[0]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

            for person in keypoints:
                img = draw_keyp(img, person, format='halpe')

            smpl = np.concatenate((img, pred_smpl, pred_side, pred_top), axis=1)
            side = np.concatenate((background, gt_smpl, gt_side, gt_top), axis=1)

            pred_smpl = np.concatenate((smpl, side), axis=0)

            if viz:
                vis_img('img', pred_smpl)

            if save_render:
                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'images')
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(os.path.join(save_path, name), pred_smpl)

            if save_mesh:
                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'meshes')
                os.makedirs(save_path, exist_ok=True)
                
                for agent in range(self.num_agent):
                    write_obj(gt_vert[agent], self.smpl_neutral.faces, os.path.join(save_path, idx + '_' + str(agent) + '_gt.obj'))
                    write_obj(vert[agent], self.smpl_neutral.faces, os.path.join(save_path, idx + '_' + str(agent) + '_pred.obj'))

            if save_params:
                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'params', idx)
                os.makedirs(save_path, exist_ok=True)
                pose_param = torch.cat([pose_0[i][None,:], pose_1[i][None,:]], dim=0)
                shape_param = torch.cat([shape_0[i][None,:], shape_1[i][None,:]], dim=0)
                trans_param = torch.cat([trans_0[i][None,:], trans_1[i][None,:]], dim=0)

                if self.train_dataset.gt_pose is not None:
                    gt_pose_param = torch.cat([gt_pose_0[i][None,:], gt_pose_1[i][None,:]], dim=0)
                    gt_shape_param = torch.cat([gt_shape_0[i][None,:], gt_shape_1[i][None,:]], dim=0)
                    gt_trans_param = torch.cat([gt_trans_0[i][None,:], gt_trans_1[i][None,:]], dim=0)
                else:
                    gt_pose_param = torch.zeros((1,72))
                    gt_shape_param = torch.zeros((1,10))
                    gt_trans_param = torch.zeros((1,3))

                data = {'pose':pose_param.detach().cpu().numpy(),
                        'trans':trans_param.detach().cpu().numpy(),
                        'betas':shape_param.detach().cpu().numpy(),
                        'gt_pose':gt_pose_param.detach().cpu().numpy(),
                        'gt_trans':gt_trans_param.detach().cpu().numpy(),
                        'gt_betas':gt_shape_param.detach().cpu().numpy(),}

                save_pkl(os.path.join(save_path, '0000.pkl'), data)

                save_path = os.path.join(self.model_path, 'smpl_epoch%04d_%s' %(epoch, m_type), 'camparams', idx)
                os.makedirs(save_path, exist_ok=True)

                intri = self.train_dataset.intrinsic
                extri = self.train_dataset.extrinsic
                save_camparam(os.path.join(save_path, 'camparams.txt'), [intri], [extri])

    def net_set(self, mode):
        assert mode in [0, 1, 2]

        self.proxemics_prior = interhuman_diffusion_phys(self.smpl_gpu, frame_length=16).cuda()

        # Load pretrain parameters
        model_dict = self.proxemics_prior.state_dict()
        params = torch.load('data/best_reconstruction.pkl')
        premodel_dict = params['model']
        premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        model_dict.update(premodel_dict)
        self.proxemics_prior.load_state_dict(model_dict)
        print("Load pretrain proxemics_prior parameters")

        self.net = POP_no_unet(
            c_geom=self.net_parms.c_geom, # channels of the geometric features
            geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            nf=self.net_parms.nf, # num filters for the unet
            hsize=self.net_parms.hsize, # hidden layer size of the ShapeDecoder MLP
            up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            uv_feat_dim=2, # input dimension of the uv coordinates
        ).cuda()
            
        self.net1 = POP_no_unet(
            c_geom=self.net_parms.c_geom, # channels of the geometric features
            geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            nf=self.net_parms.nf, # num filters for the unet
            hsize=self.net_parms.hsize, # hidden layer size of the ShapeDecoder MLP
            up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            uv_feat_dim=2, # input dimension of the uv coordinates
        ).cuda()

        geo_feature = torch.ones(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        self.geo_feature_0 = nn.Parameter(geo_feature.requires_grad_(True))
        self.geo_feature_1 = nn.Parameter(geo_feature.requires_grad_(True))
        
        if self.model_parms.train_stage == 2:
            self.pose_encoder = UnetNoCond5DS(
                input_nc=3,
                output_nc=self.net_parms.c_pose,
                nf=self.net_parms.nf,
                up_mode=self.net_parms.up_mode,
                use_dropout=False,
            ).cuda()

    def training_setup(self):
        if self.model_parms.train_stage == 1:
            self.optimizer = torch.optim.Adam(
            [
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net},
                {"params": self.net1.parameters(), "lr": self.opt_parms.lr_net},
                {"params": self.geo_feature_0, "lr": self.opt_parms.lr_geomfeat},
                {"params": self.geo_feature_1, "lr": self.opt_parms.lr_geomfeat}
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)
            self.prior_optimizer = torch.optim.Adam(
            [
                {"params": self.proxemics_prior.parameters(), "lr": 2e-5},
            ])
        else:
            self.optimizer = torch.optim.Adam(
            [   
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net * 0.1},
                {"params": self.pose_encoder.parameters(), "lr": self.opt_parms.lr_net},
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)
            self.prior_optimizer = torch.optim.Adam(
            [
                {"params": self.proxemics_prior.parameters(), "lr": 2e-5},
            ])
            
    def save_all(self, iteration, mpjpe, motion_loss):
        net_save_path = os.path.join(self.model_path, "net")
        mkdir_p(net_save_path)

        current_state = {
                "net": self.net.state_dict(),
                "proxemics_prior": self.proxemics_prior.state_dict(),
                "geo_feature_0": self.geo_feature_0,
                "geo_feature_1": self.geo_feature_1,
                "pose_0": self.pose_0.state_dict(),
                "transl_0": self.transl_0.state_dict(),
                "betas_0": self.betas_0.state_dict(),
                "pose_1": self.pose_1.state_dict(),
                "transl_1": self.transl_1.state_dict(),
                "betas_1": self.betas_1.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "prior_optimizer": self.prior_optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}

        if True:
            self.best_MPJPE_count = 0
            self.best_MPJPE = mpjpe
            self.best_MPJPE_state = current_state
            save_path = os.path.join(net_save_path, "MPJPE_%.5f_Loss_%.5f_net_%d.pth" %(mpjpe, motion_loss, iteration))
            # if self.best_MPJPE_path is not None and os.path.exists(self.best_MPJPE_path):
            #     os.remove(self.best_MPJPE_path)
            torch.save(self.best_MPJPE_state, save_path)
            self.best_MPJPE_path = save_path

    def save(self, iteration, mpjpe, motion_loss):
        net_save_path = os.path.join(self.model_path, "net")
        mkdir_p(net_save_path)

        current_state = {
                "net": self.net.state_dict(),
                "proxemics_prior": self.proxemics_prior.state_dict(),
                "geo_feature_0": self.geo_feature_0,
                "geo_feature_1": self.geo_feature_1,
                "pose_0": self.pose_0.state_dict(),
                "transl_0": self.transl_0.state_dict(),
                "betas_0": self.betas_0.state_dict(),
                "pose_1": self.pose_1.state_dict(),
                "transl_1": self.transl_1.state_dict(),
                "betas_1": self.betas_1.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "prior_optimizer": self.prior_optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}

        if mpjpe < self.best_MPJPE:
            self.best_MPJPE_count = 0
            self.best_MPJPE = mpjpe
            self.best_MPJPE_state = current_state
            save_path = os.path.join(net_save_path, "MPJPE_%.5f_Loss_%.5f_net_%d.pth" %(mpjpe, motion_loss, iteration))
            if self.best_MPJPE_path is not None and os.path.exists(self.best_MPJPE_path):
                os.remove(self.best_MPJPE_path)
            torch.save(self.best_MPJPE_state, save_path)
            self.best_MPJPE_path = save_path

        if motion_loss < self.best_opt_loss and iteration > self.opt_parms.pose_op_start_iter:
            self.best_opt_loss_count = 0
            self.best_opt_loss = motion_loss
            self.best_opt_state = current_state
            save_path = os.path.join(net_save_path, "Loss_%.5f_MPJPE_%.5f_net_%d.pth" %(motion_loss, mpjpe, iteration))
            if self.best_opt_loss_path is not None and os.path.exists(self.best_opt_loss_path):
                os.remove(self.best_opt_loss_path)
            torch.save(self.best_opt_state, save_path)
            self.best_opt_loss_path = save_path

        # if self.model_parms.train_stage == 1:
        #     torch.save(current_state, 
        #     os.path.join(net_save_path, "MPJPE_%.5f_Loss_%.5f_net_%d.pth" %(mpjpe, motion_loss, iteration)))
        # else:
        #     torch.save(
        #         {
        #         "pose_encoder": self.pose_encoder.state_dict(),
        #         "geo_feature": self.geo_feature,
        #         "pose": self.pose.state_dict(),
        #         "transl": self.transl.state_dict(),
        #         "net": self.net.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #         "scheduler": self.scheduler.state_dict()}, 
        #     os.path.join(net_save_path,  "pose_encoder.pth"))

    def load(self, checkpoint, test=False):

        saved_model_state = torch.load(checkpoint)
        print('load pth: ', checkpoint)
        self.net.load_state_dict(saved_model_state["net"], strict=False)

        if self.model_parms.train_stage  ==1:
            if not test:
                self.pose_0.load_state_dict(saved_model_state["pose_0"], strict=False)
                self.transl_0.load_state_dict(saved_model_state["transl_0"], strict=False)
                self.betas_0.load_state_dict(saved_model_state["betas_0"], strict=False)
                self.pose_1.load_state_dict(saved_model_state["pose_1"], strict=False)
                self.transl_1.load_state_dict(saved_model_state["transl_1"], strict=False)
                self.betas_1.load_state_dict(saved_model_state["betas_1"], strict=False)
            # if self.train_mode == 0:
            self.geo_feature_0.data[...] = saved_model_state["geo_feature_0"].data[...]
            self.geo_feature_1.data[...] = saved_model_state["geo_feature_1"].data[...]

        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved_model_state["optimizer"])
            self.prior_optimizer.load_state_dict(saved_model_state["prior_optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved_model_state["scheduler"])


    def stage_load(self, ckpt_path):

        saved_model_state = torch.load(ckpt_path)
        
        self.net.load_state_dict(saved_model_state["net"], strict=False)
        self.proxemics_prior.load_state_dict(saved_model_state["proxemics_prior"], strict=False)
        self.pose_0.load_state_dict(saved_model_state["pose_0"], strict=False)
        self.transl_0.load_state_dict(saved_model_state["transl_0"], strict=False)
        self.betas_0.load_state_dict(saved_model_state["betas_0"], strict=False)
        self.pose_1.load_state_dict(saved_model_state["pose_1"], strict=False)
        self.transl_1.load_state_dict(saved_model_state["transl_1"], strict=False)
        self.betas_1.load_state_dict(saved_model_state["betas_1"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature_0.data[...] = saved_model_state["geo_feature_0"].data[...]
        self.geo_feature_1.data[...] = saved_model_state["geo_feature_1"].data[...]

        if self.prior_optimizer is not None:
            self.prior_optimizer.load_state_dict(saved_model_state["prior_optimizer"])

    def stage2_load(self, epoch):
    
        pose_encoder_path = os.path.join(self.model_parms.project_path, self.model_path, "net/iteration_{}".format(epoch))

        pose_encoder_state = torch.load(
            os.path.join(pose_encoder_path, "pose_encoder.pth"))
        print('load pth: ', os.path.join(pose_encoder_path, "pose_encoder.pth"))

        self.net.load_state_dict(pose_encoder_state["net"], strict=False)
        self.pose.load_state_dict(pose_encoder_state["pose"], strict=False)
        self.transl.load_state_dict(pose_encoder_state["transl"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature.data[...] = pose_encoder_state["geo_feature"].data[...]
        self.pose_encoder.load_state_dict(pose_encoder_state["pose_encoder"], strict=False)

    def getTrainDataloader(self, g):
        return torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            num_workers = 0,
                                            worker_init_fn = worker_init_fn,
                                            drop_last = False,
                                            generator=g,)

    def getTestDataset(self,):
        self.test_dataset = MonoDataset_test(self.model_parms)
        return self.test_dataset
    
    def getNovelposeDataset(self,):
        self.novel_pose_dataset = MonoDataset_novel_pose(self.model_parms)
        return self.novel_pose_dataset

    def getNovelviewDataset(self,):
        self.novel_view_dataset = MonoDataset_novel_view(self.model_parms)
        return self.novel_view_dataset

    def zero_grad(self, epoch, kwargs):
        self.optimizer.zero_grad()

        if self.model_parms.train_stage  ==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                if kwargs.use_prior:
                    self.prior_optimizer.zero_grad()
                else:
                    self.optimizer_pose.zero_grad()
                

    def step(self, epoch, kwargs):

        self.optimizer.step()
        self.scheduler.step()
        if self.model_parms.train_stage  ==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                if kwargs.use_prior:
                    self.prior_optimizer.step()
                else:
                    self.optimizer_pose.step()
                
            
    def prior_infer(self, idx, step=0):
        pose_batch_0 = self.pose_0(idx)
        transl_batch_0 = self.transl_0(idx)
        betas_batch_0 = self.betas_0(torch.tensor([0], device=pose_batch_0.device)).expand(self.batch_size*self.frame_length, -1).cuda()
        pose_batch_1 = self.pose_1(idx)
        transl_batch_1 = self.transl_1(idx)
        betas_batch_1 = self.betas_1(torch.tensor([0], device=pose_batch_0.device)).expand(self.batch_size*self.frame_length, -1).cuda()

        pose = torch.cat([pose_batch_0[:,None], pose_batch_1[:,None]], dim=1)
        transl = torch.cat([transl_batch_0[:,None], transl_batch_1[:,None]], dim=1)
        betas = torch.cat([betas_batch_0[:,None], betas_batch_1[:,None]], dim=1)

        features = self.train_dataset.img_features[idx]
        keypoints = self.train_dataset.keypoints[idx]
        pred_keypoints = self.train_dataset.pred_keypoints[idx]
        center = self.train_dataset.crop_center[idx]
        scale = self.train_dataset.crop_scale[idx]
        img_w = self.train_dataset.img_w[idx]
        img_h = self.train_dataset.img_h[idx]
        focal_length = self.train_dataset.focal_length[idx]
        init_pose_6d = self.train_dataset.init_pose_6d[idx]

        f, n = pose.shape[:2]
        b = 1
        data = {}

        curr_pose_6d = pose.reshape(-1, 3)
        curr_pose_6d = axis_angle_to_matrix(curr_pose_6d)
        curr_pose_6d = matrix_to_rotation_6d(curr_pose_6d)
        data['curr_pose_6d'] = curr_pose_6d.reshape(b, f, n, -1)
        data['curr_transl'] = transl.reshape(b, f, n, -1)
        data['curr_betas'] = betas.reshape(b, f, n, -1)
        data['init_pose_6d'] = init_pose_6d.reshape(b, f, n, -1)
        data['features'] = features.reshape(b, f, n, -1)
        data['keypoints'] = keypoints.reshape(b*f*n, 26, 3)
        data['pred_keypoints'] = pred_keypoints.reshape(b*f*n, 26, 3)
        data['center'] = center.reshape(b*f*n, 2)
        data['scale'] = scale.reshape(b*f*n, )
        data['img_w'] = img_w.reshape(b*f*n, )
        data['img_h'] = img_h.reshape(b*f*n, )
        data['focal_length'] = focal_length.reshape(b*f*n, )
        data['single_person'] = torch.zeros((b, f)).float().cuda()
        data['imgname'] = [os.path.join(self.train_dataset.data_folder, 'images', self.train_dataset.name_list[i][1] + '.jpg') for i in idx]

        # self.proxemics_prior(data)

        updated_pose, updated_transl, updated_betas = self.proxemics_prior.proxemics_infer(data, step=step)

        self.pose_0.weight.data[idx] = updated_pose[:self.frame_length]
        self.pose_1.weight.data[idx] = updated_pose[self.frame_length:]
        self.transl_0.weight.data[idx] = updated_transl[:self.frame_length]
        self.transl_1.weight.data[idx] = updated_transl[self.frame_length:]
        self.betas_0.weight.data = updated_betas[0:1]
        self.betas_1.weight.data = updated_betas[self.frame_length:self.frame_length+1]

        return updated_pose, updated_transl, updated_betas, data

    def contact_loss(self, pred_verts, frame_num, agent_num=2):

        contact_verts = self.train_dataset.contact_verts 
        frame_idx = self.train_dataset.contact_idx 
        verts1_idx = self.train_dataset.verts1_idx 
        verts2_idx = self.train_dataset.verts2_idx 

        n_point = verts1_idx[0].shape[0]

        assert len(contact_verts) == frame_num
        pred_verts_0 = pred_verts[:frame_num][None,:]
        pred_verts_1 = pred_verts[frame_num:][None,:]

        pred_verts = torch.cat([pred_verts_0, pred_verts_1], dim=0)

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

    def reprojection_loss(self, live_smpl, pred_keypoints, center, img_w, img_h, focal_length, mask):

        joint_halpe_3d = self.smpl_gpu.J_halpe_regressor @ live_smpl.vertices
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        joint_halpe_2d = perspective_projection(joint_halpe_3d,
                                                rotation=torch.eye(3, device=joint_halpe_3d.device).unsqueeze(0).expand(joint_halpe_3d.shape[0], -1, -1),
                                                translation=torch.zeros(3, device=joint_halpe_3d.device).unsqueeze(0).expand(joint_halpe_3d.shape[0], -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        
        pred_keypoints = torch.cat([pred_keypoints[:,0], pred_keypoints[:,1]], dim=0)
        center = torch.cat([center[:,0], center[:,1]], dim=0)

        if mask is not None:
            coords = pred_keypoints[...,:2].clone().to(torch.int32)
            coords_x_ori = coords[...,0]
            coords_y_ori = coords[...,1]
            coords_x = torch.clip(coords_x_ori.clone(), 0, mask.shape[2]-1)
            coords_y = torch.clip(coords_y_ori.clone(), 0, mask.shape[1]-1)
            keyp_mask = mask[torch.arange(len(coords)).unsqueeze(1).cuda(), coords_y.to(torch.long), coords_x.to(torch.long)]
            keyp_mask[coords_x_ori > mask.shape[2]-1] = 0
            keyp_mask[coords_y_ori > mask.shape[1]-1] = 0
            keyp_mask[coords_x_ori < 0] = 0
            keyp_mask[coords_y_ori < 0] = 0

        else:
            keyp_mask = 1

        # for i, img in enumerate(imgname):
        #     if i > 0:
        #         break
        #     img = cv2.imread(img)
        #     vert = live_smpl.vertices[[i, 16+i]]
        #     renderer = Renderer(focal_length=self.train_dataset.intrinsic[0][0], center=(self.train_dataset.intrinsic[0][2], self.train_dataset.intrinsic[1][2]), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_neutral.faces, same_mesh_color=True)

        #     pred_smpl = renderer.render_front_view(vert.detach().cpu().numpy(), bg_img_rgb=img.copy())
        #     pred_side = renderer.render_side_view(vert.detach().cpu().numpy())

        #     vis_mask = mask[[i, 16+i]].detach().cpu().numpy()
        #     img = overlay_mask_on_image(img, vis_mask, alpha=0.75)

        #     mask_temp = keyp_mask[[i, 16+i]]
        #     pred_keyp = pred_keypoints[[i, 16+i]]
        #     for p_id, keyp in enumerate(pred_keyp.detach().cpu().numpy()):
        #         # if p_id == 0:
        #         #     continue
        #         for k_i, kp in enumerate(keyp[:,:2].astype(np.int)):
        #             if mask_temp[p_id, k_i] > 0.5:
        #                 img = cv2.circle(img, tuple(kp), 5, (0,0,255), -1)
        #             else:
        #                 img = cv2.circle(img, tuple(kp), 5, (255,0,0), -1)
        #     pred_keyp = joint_halpe_2d[[i, 16+i]]
        #     for p_id, keyp in enumerate(pred_keyp.detach().cpu().numpy()):
        #         # if p_id == 0:
        #         #     continue
        #         for k_i, kp in enumerate(keyp[:,:2].astype(np.int)):
        #             img = cv2.circle(img, tuple(kp), 5, (0,255,255), -1)

        #     smpl = np.concatenate((img, pred_smpl, pred_side), axis=1)

        #     vis_img('smpl', smpl)
        
        joint_halpe_2d = (joint_halpe_2d - center[:,None,:]) / 256
        pred_keypoints[...,:2] = (pred_keypoints[...,:2] - center[:,None,:]) / 256

        pred_keypoints[...,2][pred_keypoints[...,2] < 0.7] = 0.
        pred_keypoints[...,2] = pred_keypoints[...,2] * keyp_mask
        loss_2d = (pred_keypoints[...,2] * ((joint_halpe_2d - pred_keypoints[...,:2])**2).sum(dim=-1)).mean()

        return loss_2d

    def smoothness_loss(self, live_smpl, length):

        # # smoothness
        # vertices_0 = live_smpl.vertices[:length]
        # vertices_1 = live_smpl.vertices[length:]
        # joint_halpe_3d_0 = self.smpl_gpu.J_halpe_regressor @ vertices_0
        # joint_halpe_3d_1 = self.smpl_gpu.J_halpe_regressor @ vertices_1
        # joint_halpe_3d = torch.cat([joint_halpe_3d_0, joint_halpe_3d_1], dim=1)
        # weights = torch.ones((1,26*2), dtype=torch.float32, device=torch.device('cuda'))
        # # weights[:,[7,8,9,10,13,14,15,16,20,21,22,23,24,25],:] = 0.5
        # vel_halpe_3d = torch.linalg.norm(joint_halpe_3d[1:] - joint_halpe_3d[:-1], dim=-1) * weights
        # smoothness_loss = vel_halpe_3d.mean()

        # smoothness (2nd-order)
        vertices_0 = live_smpl.vertices[:length]
        vertices_1 = live_smpl.vertices[length:]
        joint_halpe_3d_0 = self.smpl_gpu.J_halpe_regressor @ vertices_0
        joint_halpe_3d_1 = self.smpl_gpu.J_halpe_regressor @ vertices_1
        joint_halpe_3d = torch.cat([joint_halpe_3d_0, joint_halpe_3d_1], dim=1)  # [T, J, 3]

        # Compute second-order differences (acceleration)
        acc_halpe_3d = joint_halpe_3d[2:] - 2 * joint_halpe_3d[1:-1] + joint_halpe_3d[:-2]

        # Optional: apply weights (if you want per-joint weights)
        weights = torch.ones((1, 26 * 2 * 3), dtype=torch.float32, device='cuda')  # shape: [1, 52]
        # weights[:,[7,8,9,10,13,14,15,16,20,21,22,23,24,25]] = 0.5
        acc_halpe_3d = acc_halpe_3d.reshape(acc_halpe_3d.shape[0], -1)  # [T-2, J*3]
        weighted_acc = acc_halpe_3d * weights  # broadcasting weights over [T-2, J*3]

        smoothness_loss = torch.linalg.norm(weighted_acc, dim=-1).mean()

        return smoothness_loss

    def regularization_loss(self, pose_batch, betas_batch, transl_batch):

        # init prior loss
        pred_pose_6d = axis_angle_to_matrix(pose_batch.reshape(-1, 3))
        pred_pose_6d = matrix_to_rotation_6d(pred_pose_6d).reshape(-1, 144)

        init_pose_0 = self.init_pose[:,0].reshape(-1, 72)
        init_pose_1 = self.init_pose[:,1].reshape(-1, 72)
        init_pose_batch = torch.cat([init_pose_0, init_pose_1], dim=0)
        init_pose_6d = axis_angle_to_matrix(init_pose_batch.reshape(-1, 3))
        init_pose_6d = matrix_to_rotation_6d(init_pose_6d).reshape(-1, 144)

        motion_prior = (((pred_pose_6d - init_pose_6d)**2).sum(-1)).mean()


        init_transl_0 = self.init_transl[:,0].reshape(-1, 3)
        init_transl_1 = self.init_transl[:,1].reshape(-1, 3)
        init_transl_batch = torch.cat([init_transl_0, init_transl_1], dim=0)

        transl_prior = (((transl_batch - init_transl_batch)**2).sum(-1)).mean()


        idx_0 = torch.tensor(([0]), dtype=torch.int64, device=pose_batch.device)
        betas = torch.cat([self.betas_0(idx_0), self.betas_1(idx_0)], dim=0)

        shape_prior = (((betas - self.init_shapes)**2).sum(-1)).mean()

        return motion_prior, shape_prior, transl_prior

    def penetration_loss(self, live_smpl, length):

        # penetration loss
        verts = torch.cat([live_smpl.vertices[:length][:,None], live_smpl.vertices[length:][:,None]], dim=1).reshape(-1,6890,3)
        batch_size = verts.shape[0]
        vertices = verts
        face_tensor = torch.tensor(self.smpl_gpu.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([batch_size,
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        with torch.no_grad():
            if print_timings:
                start = time.time()
            collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

            if False:
                if print_timings:
                    start = time.time()
                collision_idxs = self.filter_faces(collision_idxs)
                if print_timings:
                    torch.cuda.synchronize()
                    print('Collision filtering: {:5f}ms'.format((time.time() -
                                                                start) * 1000))

        if print_timings:
                start = time.time()
        pen_loss = self.pen_distance(triangles, collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

        filtered_pen_loss = pen_loss[pen_loss > self.pen_threshold].mean() / 1000.

        if torch.isnan(filtered_pen_loss):
            filtered_pen_loss = 0.

        return filtered_pen_loss


    def optimization(self, batch_data, iteration, epoch, kwargs):

        rendered_images = []
        length = self.pose_0.num_embeddings
        all_idx = torch.arange(0, length).cuda()
        idx = batch_data['pose_idx']

        # prior
        use_prior = kwargs.use_prior
        if use_prior and epoch > self.opt_parms.pose_op_start_iter:
            pose_batch, transl_batch, betas_batch, data = self.prior_infer(idx, step=step)

            pred_keypoints = data['pred_keypoints'].reshape(self.frame_length, self.num_agent, 26, 3)
            center = data['center'].reshape(self.frame_length, self.num_agent, 2)
            img_w = data['img_w']
            img_h = data['img_h']
            focal_length = data['focal_length']
            imgname = data['imgname']
        else:
            pred_keypoints = self.train_dataset.pred_keypoints
            center = self.train_dataset.crop_center
            img_w = self.train_dataset.img_w.reshape(-1,)
            img_h = self.train_dataset.img_h.reshape(-1,)
            focal_length = self.train_dataset.focal_length.reshape(-1,)
            imgname = [os.path.join(self.train_dataset.data_folder, 'images', self.train_dataset.name_list[i][1] + '.jpg') for i in all_idx]

            pose_batch_0 = self.pose_0(all_idx)
            transl_batch_0 = self.transl_0(all_idx)
            betas_batch_0 = self.betas_0(torch.tensor([0], device=pose_batch_0.device)).expand(all_idx.shape[0], -1).cuda()
            pose_batch_1 = self.pose_1(all_idx)
            transl_batch_1 = self.transl_1(all_idx)
            betas_batch_1 = self.betas_1(torch.tensor([0], device=pose_batch_0.device)).expand(all_idx.shape[0], -1).cuda()

            pose_batch = torch.cat([pose_batch_0, pose_batch_1], dim=0)
            transl_batch = torch.cat([transl_batch_0, transl_batch_1], dim=0)
            betas_batch = torch.cat([betas_batch_0, betas_batch_1], dim=0)

        live_smpl = self.smpl_model.forward(betas=betas_batch[:,:10],
                            global_orient=pose_batch[:, :3],
                            transl = transl_batch,
                            body_pose=pose_batch[:, 3:])

        self.loss['reproj'] = self.reprojection_loss(live_smpl, pred_keypoints, center, img_w, img_h, focal_length, None)

        self.loss['smoothness'] = self.smoothness_loss(live_smpl, length)

        self.loss['motion_prior'], self.loss['shape_prior'], self.loss['transl_prior'] = self.regularization_loss(pose_batch, betas_batch, transl_batch)

        self.loss['pen_loss'] = self.penetration_loss(live_smpl, length)

        if self.train_dataset.contact_verts is not None:
            self.loss['contact_loss'] = self.contact_loss(live_smpl.vertices, length)

        prior_loss = self.loss['reproj'] * self.loss_weights['reproj'] + self.loss['motion_prior'] * self.loss_weights['motion_prior'] + self.loss['shape_prior'] * self.loss_weights['shape_prior'] + self.loss['transl_prior'] * self.loss_weights['transl_prior'] + self.loss['smoothness'] * self.loss_weights['smoothness'] + self.loss['pen_loss'] * self.loss_weights['pen_loss'] + self.loss['contact_loss'] * self.loss_weights['contact_loss'] + torch.tensor((0), dtype=torch.float32).cuda()

        use_appearance = kwargs.use_appearance
        if use_appearance:
            A_0 = live_smpl.A[:length][idx]
            A_1 = live_smpl.A[length:][idx]
            A = torch.cat([A_0, A_1], dim=0)

            cano2live_jnt_mats = torch.matmul(A, self.inv_mats)

            # Optimizable tensors
            geom_featmap_0 = self.geo_feature_0
            geom_featmap_1 = self.geo_feature_1

            geom_featmap = torch.cat([geom_featmap_0, geom_featmap_1], dim=0)
            uv_coord_map = self.uv_coord_map.expand(self.num_agent, -1, -1).contiguous()

            # U-Net
            pred_res, pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                        geom_featmap=geom_featmap[:1],
                                                        uv_loc=uv_coord_map[:1])

            pred_res1, pred_scales1, pred_shs1, = self.net1.forward(pose_featmap=None,
                                                        geom_featmap=geom_featmap[1:],
                                                        uv_loc=uv_coord_map[1:])

            pred_res = torch.cat([pred_res, pred_res1], dim=0)
            pred_scales = torch.cat([pred_scales, pred_scales1], dim=0)
            pred_shs = torch.cat([pred_shs, pred_shs1], dim=0)

            # temp code
            texture_map = pred_shs.detach()

            pred_res_0, pred_scales_0, pred_shs_0 = pred_res[:1].expand(self.batch_size*self.frame_length, -1, -1), pred_scales[:1].expand(self.batch_size*self.frame_length, -1, -1), pred_shs[:1].expand(self.batch_size*self.frame_length, -1, -1)
            pred_res_1, pred_scales_1, pred_shs_1 = pred_res[1:].expand(self.batch_size*self.frame_length, -1, -1), pred_scales[1:].expand(self.batch_size*self.frame_length, -1, -1), pred_shs[1:].expand(self.batch_size*self.frame_length, -1, -1)

            pred_res = torch.cat([pred_res_0, pred_res_1], dim=0)
            pred_scales = torch.cat([pred_scales_0, pred_scales_1], dim=0)
            pred_shs = torch.cat([pred_shs_0, pred_shs_1], dim=0)

            pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
            pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

            cano_deform_point = pred_point_res + self.query_points

            pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
            full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

            if iteration < 1000:
                pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
            else:
                pred_scales = pred_scales.permute([0,2,1])

            pred_shs = pred_shs.permute([0,2,1])

            pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
            pred_scales = pred_scales.repeat(1,1,3)

            pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()


            offset_loss = torch.mean(pred_res ** 2)
            geo_loss = torch.mean(self.geo_feature_0**2)
            geo_loss += torch.mean(self.geo_feature_1**2)
            scale_loss = torch.mean(pred_scales)

            full_pred_0 = full_pred[:self.batch_size*self.frame_length]
            pred_shs_0 = pred_shs[:self.batch_size*self.frame_length]
            pred_scales_0 = pred_scales[:self.batch_size*self.frame_length]
            full_pred_1 = full_pred[self.batch_size*self.frame_length:]
            pred_shs_1 = pred_shs[self.batch_size*self.frame_length:]
            pred_scales_1 = pred_scales[self.batch_size*self.frame_length:]

            # print('point', full_pred[0,0].detach().cpu().numpy())

            for batch_index in range(self.frame_length):
                FovX = batch_data['FovX'][batch_index]
                FovY = batch_data['FovY'][batch_index]
                height = batch_data['height'][batch_index]
                width = batch_data['width'][batch_index]
                world_view_transform = batch_data['world_view_transform'][batch_index]
                full_proj_transform = batch_data['full_proj_transform'][batch_index]
                camera_center = batch_data['camera_center'][batch_index]
            
                points_0 = full_pred_0[batch_index]
                colors_0 = pred_shs_0[batch_index]
                scales_0 = pred_scales_0[batch_index]

                points_1 = full_pred_1[batch_index]
                colors_1 = pred_shs_1[batch_index]
                scales_1 = pred_scales_1[batch_index]

                points = torch.cat([points_0, points_1], dim=0)
                colors = torch.cat([colors_0, colors_1], dim=0)
                scales = torch.cat([scales_0, scales_1], dim=0)

                rendered_images.append(
                    render_batch(
                        points=points,
                        shs=None,
                        colors_precomp=colors,
                        rotations=self.fix_rotation,
                        scales=scales,
                        opacity=self.fix_opacity,
                        FovX=FovX,
                        FovY=FovY,
                        height=height,
                        width=width,
                        bg_color=self.background,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,
                        active_sh_degree=0,
                        camera_center=camera_center
                    )
                )

        if use_appearance:
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, scale_loss, prior_loss, texture_map
        else:
            return None, None, 0., 0., 0., prior_loss, None

        pass

    def train_stage1(self, batch_data, iteration, epoch, kwargs):

        rendered_images = []
        idx = batch_data['pose_idx']
        mask = torch.cat([batch_data['mask_0'], batch_data['mask_1']], dim=0)

        step = iteration

        # while True:
        # prior
        use_prior = kwargs.use_prior
        if use_prior and epoch > self.opt_parms.pose_op_start_iter:
            pose_batch, transl_batch, betas_batch, data = self.prior_infer(idx, step=step)

            pred_keypoints = data['pred_keypoints'].reshape(self.frame_length, self.num_agent, 26, 3)
            center = data['center'].reshape(self.frame_length, self.num_agent, 2)
            img_w = data['img_w']
            img_h = data['img_h']
            focal_length = data['focal_length']
            imgname = data['imgname']
        else:
            pred_keypoints = self.train_dataset.pred_keypoints[idx].reshape(self.frame_length, self.num_agent, 26, 3)
            center = self.train_dataset.crop_center[idx].reshape(self.frame_length, self.num_agent, 2)
            img_w = self.train_dataset.img_w[idx].reshape(-1,)
            img_h = self.train_dataset.img_h[idx].reshape(-1,)
            focal_length = self.train_dataset.focal_length[idx].reshape(-1,)
            imgname = [os.path.join(self.train_dataset.data_folder, 'images', self.train_dataset.name_list[i][1] + '.jpg') for i in idx]

            pose_batch_0 = self.pose_0(idx)
            transl_batch_0 = self.transl_0(idx)
            betas_batch_0 = self.betas_0(torch.tensor([0], device=pose_batch_0.device)).expand(self.batch_size*self.frame_length, -1).cuda()
            pose_batch_1 = self.pose_1(idx)
            transl_batch_1 = self.transl_1(idx)
            betas_batch_1 = self.betas_1(torch.tensor([0], device=pose_batch_0.device)).expand(self.batch_size*self.frame_length, -1).cuda()

            pose_batch = torch.cat([pose_batch_0, pose_batch_1], dim=0)
            transl_batch = torch.cat([transl_batch_0, transl_batch_1], dim=0)
            betas_batch = torch.cat([betas_batch_0, betas_batch_1], dim=0)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=betas_batch[:,:10],
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        use_appearance = kwargs.use_appearance
        if use_appearance:
            # Optimizable tensors
            geom_featmap_0 = self.geo_feature_0
            geom_featmap_1 = self.geo_feature_1

            geom_featmap = torch.cat([geom_featmap_0, geom_featmap_1], dim=0)
            uv_coord_map = self.uv_coord_map.expand(self.num_agent, -1, -1).contiguous()

            # U-Net
            pred_res, pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map)

            # temp code
            texture_map = pred_shs.detach()

            pred_res_0, pred_scales_0, pred_shs_0 = pred_res[:1].expand(self.batch_size*self.frame_length, -1, -1), pred_scales[:1].expand(self.batch_size*self.frame_length, -1, -1), pred_shs[:1].expand(self.batch_size*self.frame_length, -1, -1)
            pred_res_1, pred_scales_1, pred_shs_1 = pred_res[1:].expand(self.batch_size*self.frame_length, -1, -1), pred_scales[1:].expand(self.batch_size*self.frame_length, -1, -1), pred_shs[1:].expand(self.batch_size*self.frame_length, -1, -1)

            pred_res = torch.cat([pred_res_0, pred_res_1], dim=0)
            pred_scales = torch.cat([pred_scales_0, pred_scales_1], dim=0)
            pred_shs = torch.cat([pred_shs_0, pred_shs_1], dim=0)

            pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
            pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

            cano_deform_point = pred_point_res + self.query_points

            pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
            full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

            if iteration < 1000:
                pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
            else:
                pred_scales = pred_scales.permute([0,2,1])

            pred_shs = pred_shs.permute([0,2,1])

            pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
            pred_scales = pred_scales.repeat(1,1,3)

            pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()


        loss_2d = 0.
        motion_prior = 0.
        transl_prior = 0.
        shape_prior = 0
        smoothness_loss = 0.
        filtered_pen_loss = 0.
        if epoch > self.opt_parms.pose_op_start_iter:

            # re-projection loss
            joint_halpe_3d = self.smpl_gpu.J_halpe_regressor @ live_smpl.vertices
            camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
            joint_halpe_2d = perspective_projection(joint_halpe_3d,
                                                    rotation=torch.eye(3, device=joint_halpe_3d.device).unsqueeze(0).expand(joint_halpe_3d.shape[0], -1, -1),
                                                    translation=torch.zeros(3, device=joint_halpe_3d.device).unsqueeze(0).expand(joint_halpe_3d.shape[0], -1),
                                                    focal_length=focal_length,
                                                    camera_center=camera_center)
            pred_keypoints = torch.cat([pred_keypoints[:,0], pred_keypoints[:,1]], dim=0)

            coords = pred_keypoints[...,:2].clone().to(torch.int32)
            coords_x_ori = coords[...,0]
            coords_y_ori = coords[...,1]
            coords_x = torch.clip(coords_x_ori.clone(), 0, mask.shape[2]-1)
            coords_y = torch.clip(coords_y_ori.clone(), 0, mask.shape[1]-1)
            keyp_mask = mask[torch.arange(len(coords)).unsqueeze(1).cuda(), coords_y.to(torch.long), coords_x.to(torch.long)]
            keyp_mask[coords_x_ori > mask.shape[2]-1] = 0
            keyp_mask[coords_y_ori > mask.shape[1]-1] = 0
            keyp_mask[coords_x_ori < 0] = 0
            keyp_mask[coords_y_ori < 0] = 0

            # for i, img in enumerate(imgname):
            #     if i > 0:
            #         break
            #     img = cv2.imread(img)
            #     vert = live_smpl.vertices[[i, 16+i]]
            #     renderer = Renderer(focal_length=self.train_dataset.intrinsic[0][0], center=(self.train_dataset.intrinsic[0][2], self.train_dataset.intrinsic[1][2]), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_neutral.faces, same_mesh_color=True)

            #     pred_smpl = renderer.render_front_view(vert.detach().cpu().numpy(), bg_img_rgb=img.copy())
            #     pred_side = renderer.render_side_view(vert.detach().cpu().numpy())

            #     vis_mask = mask[[i, 16+i]].detach().cpu().numpy()
            #     img = overlay_mask_on_image(img, vis_mask, alpha=0.75)

            #     mask_temp = keyp_mask[[i, 16+i]]
            #     pred_keyp = pred_keypoints[[i, 16+i]]
            #     for p_id, keyp in enumerate(pred_keyp.detach().cpu().numpy()):
            #         # if p_id == 0:
            #         #     continue
            #         for k_i, kp in enumerate(keyp[:,:2].astype(np.int)):
            #             if mask_temp[p_id, k_i] > 0.5:
            #                 img = cv2.circle(img, tuple(kp), 5, (0,0,255), -1)
            #             else:
            #                 img = cv2.circle(img, tuple(kp), 5, (255,0,0), -1)
            #     pred_keyp = joint_halpe_2d[[i, 16+i]]
            #     for p_id, keyp in enumerate(pred_keyp.detach().cpu().numpy()):
            #         # if p_id == 0:
            #         #     continue
            #         for k_i, kp in enumerate(keyp[:,:2].astype(np.int)):
            #             img = cv2.circle(img, tuple(kp), 5, (0,255,255), -1)

            #     smpl = np.concatenate((img, pred_smpl, pred_side), axis=1)

            #     vis_img('smpl', smpl)

            center = torch.cat([center[:,0], center[:,1]], dim=0)

            joint_halpe_2d = (joint_halpe_2d - center[:,None,:]) / 256
            pred_keypoints[...,:2] = (pred_keypoints[...,:2] - center[:,None,:]) / 256

            pred_keypoints[...,2][pred_keypoints[...,2] < 0.7] = 0.
            pred_keypoints[...,2] = pred_keypoints[...,2] * keyp_mask
            loss_2d = (pred_keypoints[...,2] * ((joint_halpe_2d - pred_keypoints[...,:2])**2).sum(dim=-1)).mean()

            # init prior loss
            l1_loss = nn.L1Loss()
            l1_loss_non_red = nn.L1Loss(reduction='none')
            pose_6d = axis_angle_to_matrix(pose_batch.reshape(-1, 3))
            pose_6d = matrix_to_rotation_6d(pose_6d).reshape(-1, 144)

            init_pose_0 = self.init_pose[:,0][idx].reshape(-1, 72)
            init_pose_1 = self.init_pose[:,1][idx].reshape(-1, 72)
            init_transl_0 = self.init_transl[:,0][idx].reshape(-1, 3)
            init_transl_1 = self.init_transl[:,1][idx].reshape(-1, 3)
            init_pose_batch = torch.cat([init_pose_0, init_pose_1], dim=0)
            init_transl_batch = torch.cat([init_transl_0, init_transl_1], dim=0)

            weights = self.prior_weight[idx]
            weights = torch.cat([weights[:,0], weights[:,1]], dim=0)

            init_pose_6d = axis_angle_to_matrix(init_pose_batch.reshape(-1, 3))
            init_pose_6d = matrix_to_rotation_6d(init_pose_6d).reshape(-1, 144)

            motion_prior = (weights * l1_loss_non_red(pose_6d, init_pose_6d).mean(dim=-1)).mean()
            transl_prior = (weights * l1_loss_non_red(transl_batch, init_transl_batch).mean(dim=-1)).mean()

            idx_0 = torch.tensor(([0]), dtype=torch.int64, device=pose_6d.device)

            betas = torch.cat([self.betas_0(idx_0), self.betas_1(idx_0)], dim=0)

            shape_prior = l1_loss(betas, self.init_shapes)

            # smoothness
            all_idx = torch.arange(0, self.pose_0.num_embeddings).cuda()
            live_smpl_0 = self.smpl_model.forward(betas=self.betas_0(torch.tensor([0], device=all_idx.device)).expand(all_idx.shape[0], -1),
                                global_orient=self.pose_0(all_idx)[:, :3],
                                transl = self.transl_0(all_idx),
                                body_pose=self.pose_0(all_idx)[:, 3:])
            live_smpl_1 = self.smpl_model.forward(betas=self.betas_1(torch.tensor([0], device=all_idx.device)).expand(all_idx.shape[0], -1),
                                global_orient=self.pose_1(all_idx)[:, :3],
                                transl = self.transl_1(all_idx),
                                body_pose=self.pose_1(all_idx)[:, 3:])
            joint_halpe_3d_0 = self.smpl_gpu.J_halpe_regressor @ live_smpl_0.vertices
            joint_halpe_3d_1 = self.smpl_gpu.J_halpe_regressor @ live_smpl_1.vertices
            joint_halpe_3d = torch.cat([joint_halpe_3d_0, joint_halpe_3d_1], dim=1)
            weights = torch.ones((1,26*2), dtype=torch.float32, device=torch.device('cuda'))
            # weights[:,[7,8,9,10,13,14,15,16,20,21,22,23,24,25],:] = 0.5
            vel_halpe_3d = torch.linalg.norm(joint_halpe_3d[1:] - joint_halpe_3d[:-1], dim=-1) * weights
            smoothness_loss = vel_halpe_3d.mean()

            # pose_0 = self.pose_0.weight.detach().cpu()
            # pose_1 = self.pose_1.weight.detach().cpu()

            # t = np.arange(pose_0.shape[0])
            # min_cutoff = 0.004
            # beta = 10 #0.7

            # rotation_matrix = axis_angle_to_matrix(pose_0.reshape(-1, 3))
            # rotation_6d = matrix_to_rotation_6d(rotation_matrix)
            # rotation_6d = rotation_6d.reshape(pose_0.shape[0], -1).detach().numpy()

            # filterdata = np.zeros_like(rotation_6d)
            # filterdata[0] = rotation_6d[0]
            # one_euro_filter = OneEuroFilter(
            #     t[0], rotation_6d[0],
            #     min_cutoff=min_cutoff,
            #     beta=beta
            # )
            # for i in range(1, len(t)):
            #     filterdata[i] = one_euro_filter(t[i], rotation_6d[i])

            # smooth_pose_0 = torch.from_numpy(filterdata.reshape(pose_0.shape[0], -1)).float().cuda()

            # rotation_matrix = axis_angle_to_matrix(pose_1.reshape(-1, 3))
            # rotation_6d = matrix_to_rotation_6d(rotation_matrix)
            # rotation_6d = rotation_6d.reshape(pose_1.shape[0], -1).detach().numpy()

            # filterdata = np.zeros_like(rotation_6d)
            # filterdata[0] = rotation_6d[0]
            # one_euro_filter = OneEuroFilter(
            #     t[0], rotation_6d[0],
            #     min_cutoff=min_cutoff,
            #     beta=beta
            # )
            # for i in range(1, len(t)):
            #     filterdata[i] = one_euro_filter(t[i], rotation_6d[i])

            # smooth_pose_1 = torch.from_numpy(filterdata.reshape(pose_1.shape[0], -1)).float().cuda()

            # smooth_pose_0 = smooth_pose_0[idx]
            # smooth_pose_1 = smooth_pose_1[idx]
            # smooth_pose = torch.cat([smooth_pose_0, smooth_pose_1], dim=0)

            # smoothness_loss = l1_loss(pose_6d, smooth_pose)

            
            # prior_loss = motion_prior * 0.01 + transl_prior * 0.01 + shape_prior * 0.1 + smoothness_loss * 0.01 + filtered_pen_loss * 0.000001 + loss_2d
            # self.prior_optimizer.zero_grad()
            # prior_loss.backward()
            # self.prior_optimizer.step()
            # self.eval_smpl(0, prior_loss.detach().cpu().numpy())

        # for id in idx:
        #     if id == 0 or id == self.init_pose.shape[0] - 1:
        #         continue
        #     id = torch.tensor([id-1, id, id+1], dtype=id.dtype, device=id.device)
        #     m_pose_0 = self.pose_0(id)
        #     m_transl_0 = self.transl_0(id)
        #     m_betas_0 = self.betas_0(idx_0).expand(m_pose_0.shape[0], -1)
        #     m_pose_1 = self.pose_1(id)
        #     m_transl_1 = self.transl_1(id)
        #     m_betas_1 = self.betas_1(idx_0).expand(m_pose_0.shape[0], -1)

        #     m_pose = torch.cat([m_pose_0, m_pose_1], dim=0)
        #     m_transl = torch.cat([m_transl_0, m_transl_1], dim=0)
        #     m_betas = torch.cat([m_betas_0, m_betas_1], dim=0)

        #     verts, joints = self.smpl_gpu(m_betas, m_pose, m_transl)

        #     joints_0_med = (joints[0] + joints[2]) / 2.
        #     smoothness_loss += l1_loss(joints[1], joints_0_med)

        #     joints_1_med = (joints[3] + joints[5]) / 2.
        #     smoothness_loss += l1_loss(joints[4], joints_1_med)


            # penetration loss
            # verts = torch.cat([live_smpl.vertices[:self.frame_length][:,None], live_smpl.vertices[self.frame_length:][:,None]], dim=1).reshape(-1,6890,3)
            # batch_size = verts.shape[0]
            # vertices = verts
            # face_tensor = torch.tensor(self.smpl_gpu.faces.astype(np.int64), dtype=torch.long,
            #                         device=vertices.device).unsqueeze_(0).repeat([batch_size,
            #                                                                 1, 1])
            # bs, nv = vertices.shape[:2] # nv: 6890
            # bs, nf = face_tensor.shape[:2] # nf: 13776
            # faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
            # faces_idx = faces_idx.reshape(bs // 2, -1, 3)
            # triangles = vertices.view([-1, 3])[faces_idx]

            # print_timings = False
            # with torch.no_grad():
            #     if print_timings:
            #         start = time.time()
            #     collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
            #     if print_timings:
            #         torch.cuda.synchronize()
            #         print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

            #     if False:
            #         if print_timings:
            #             start = time.time()
            #         collision_idxs = self.filter_faces(collision_idxs)
            #         if print_timings:
            #             torch.cuda.synchronize()
            #             print('Collision filtering: {:5f}ms'.format((time.time() -
            #                                                         start) * 1000))

            # if print_timings:
            #         start = time.time()
            # pen_loss = self.pen_distance(triangles, collision_idxs)
            # if print_timings:
            #     torch.cuda.synchronize()
            #     print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

            # filtered_pen_loss = pen_loss[pen_loss > self.pen_threshold].mean()

            # if torch.isnan(filtered_pen_loss):
            #     filtered_pen_loss = 0.

        prior_loss = motion_prior * self.loss_weights['motion_prior'] + transl_prior * self.loss_weights['transl_prior'] + shape_prior * self.loss_weights['shape_prior'] + smoothness_loss * self.loss_weights['smoothness'] + filtered_pen_loss * self.loss_weights['pen_loss'] + loss_2d * self.loss_weights['reproj'] + torch.tensor((0), dtype=torch.float32).cuda()

        if use_appearance:
            offset_loss = torch.mean(pred_res ** 2)
            geo_loss = torch.mean(self.geo_feature_0**2)
            geo_loss += torch.mean(self.geo_feature_1**2)
            scale_loss = torch.mean(pred_scales)

            full_pred_0 = full_pred[:self.batch_size*self.frame_length]
            pred_shs_0 = pred_shs[:self.batch_size*self.frame_length]
            pred_scales_0 = pred_scales[:self.batch_size*self.frame_length]
            full_pred_1 = full_pred[self.batch_size*self.frame_length:]
            pred_shs_1 = pred_shs[self.batch_size*self.frame_length:]
            pred_scales_1 = pred_scales[self.batch_size*self.frame_length:]

            # print('point', full_pred[0,0].detach().cpu().numpy())

            for batch_index in range(self.frame_length):
                FovX = batch_data['FovX'][batch_index]
                FovY = batch_data['FovY'][batch_index]
                height = batch_data['height'][batch_index]
                width = batch_data['width'][batch_index]
                world_view_transform = batch_data['world_view_transform'][batch_index]
                full_proj_transform = batch_data['full_proj_transform'][batch_index]
                camera_center = batch_data['camera_center'][batch_index]
            
                points_0 = full_pred_0[batch_index]
                colors_0 = pred_shs_0[batch_index]
                scales_0 = pred_scales_0[batch_index]

                points_1 = full_pred_1[batch_index]
                colors_1 = pred_shs_1[batch_index]
                scales_1 = pred_scales_1[batch_index]

                points = torch.cat([points_0, points_1], dim=0)
                colors = torch.cat([colors_0, colors_1], dim=0)
                scales = torch.cat([scales_0, scales_1], dim=0)

                rendered_images.append(
                    render_batch(
                        points=points,
                        shs=None,
                        colors_precomp=colors,
                        rotations=self.fix_rotation,
                        scales=scales,
                        opacity=self.fix_opacity,
                        FovX=FovX,
                        FovY=FovY,
                        height=height,
                        width=width,
                        bg_color=self.background,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,
                        active_sh_degree=0,
                        camera_center=camera_center
                    )
                )

        if use_appearance:
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, scale_loss, prior_loss, texture_map
        else:
            return None, None, 0., 0., 0., prior_loss, None


    def train_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points
        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        pose_loss = torch.mean(pose_featmap ** 2)
        scale_loss = torch.mean(pred_scales)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]
        
            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0), full_pred, pose_loss, offset_loss,
        


    def render_free_stage1(self, batch_data, iteration):
        
        rendered_images = []
        pose_data = batch_data['pose_data']
        transl_data = batch_data['transl_data']

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = pose_data[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=pose_data[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()


        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index] 
        
            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0)


    def render_free_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]


        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
        # aiap_all_loss = 0
        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]

            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0)
