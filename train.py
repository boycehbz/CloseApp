import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import lpips
import torchvision
import open3d as o3d
import sys
from pytorch_msssim import ssim
from tqdm import tqdm
from utils.loss_utils import l1_loss_w
from utils.general_utils import safe_state
from utils.module_utils import set_seed
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, NetworkParams
from model.avatar_model import AvatarModel
from utils.general_utils import to_cuda, adjust_loss_weights
import time
from utils.module_utils import prepare_output_and_logger

# sys.argv = ['', '-s=data/preprocess_data/04305', '-m=output/04305', '--train_stage=1', '--save_render', '--use_appearance', '--save_params']

def train(model, net, opt, kwargs):
    seed = 7
    g = set_seed(seed)

    tb_writer = prepare_output_and_logger(model, net, opt, kwargs.checkpoint)
    avatarmodel = AvatarModel(model, net, opt, train=True)
    
    
    # avatarmodel.vis_smpl(viz=False, save_render=True, save_mesh=False, save_params=True, epoch=0, smooth=False)
    # avatarmodel.vis_smpl_frame(viz=False, save_render=True, save_mesh=True, epoch=0, smooth=False)

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    train_loader = avatarmodel.getTrainDataloader(g)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    first_iter = 0
    epoch_start = 0
    data_length = len(train_loader)
    avatarmodel.training_setup()

    if kwargs.checkpoint is not None:
        avatarmodel.load(kwargs.checkpoint)

    # if checkpoint_epochs:
    #     avatarmodel.load(checkpoint_epochs[0])
    #     epoch_start += checkpoint_epochs[0]
    #     first_iter += epoch_start * data_length

    if model.train_stage == 2:
        model.stage1_out_path = '/media/buzhenhuang/HDD/NeurIPS2024_InterGaussian-results/output/WeChat_20240501221554_full/05.20-22h07m45s/net/MPJPE_9999999.00000_Loss_2.87085_net_200.pth'
        avatarmodel.stage_load(model.stage1_out_path)
    
    progress_bar = tqdm(range(first_iter, data_length * opt.epochs), desc="Training progress")
    ema_loss_for_log = 0.0

    avatarmodel.eval_smpl(0., 0.)
    torch.cuda.empty_cache()
    # avatarmodel.vis_smpl(viz=False, save_render=True, save_mesh=True, epoch=0)

    for epoch in range(epoch_start + 1, opt.epochs + 1):

        if model.train_stage ==1:
            avatarmodel.net.train()
            avatarmodel.pose_0.train()
            avatarmodel.transl_0.train()
            avatarmodel.betas_0.train()
            avatarmodel.pose_1.train()
            avatarmodel.transl_1.train()
            avatarmodel.betas_1.train()
            avatarmodel.proxemics_prior.train()
        else:
            avatarmodel.net.train()
            avatarmodel.pose_0.eval()
            avatarmodel.transl_0.eval()
            avatarmodel.betas_0.eval()
            avatarmodel.pose_1.eval()
            avatarmodel.transl_1.eval()
            avatarmodel.betas_1.eval()
            avatarmodel.proxemics_prior.eval()
            avatarmodel.pose_encoder.train()
        
        iter_start.record()

        wdecay_rgl = adjust_loss_weights(opt.lambda_rgl, epoch, mode='decay', start=epoch_start, every=20)

        epoch_start = time.time()
        t_loss, t_motion = 0., 0.
        for _, batch_data in enumerate(train_loader):
            first_iter += 1
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']
            
            # time_start = time.time()

            if model.train_stage == 1:

                # image, points, offset_loss, geo_loss, scale_loss, motion_loss, texture_map = avatarmodel.train_stage1(batch_data, first_iter, epoch, kwargs)
                image, points, offset_loss, geo_loss, scale_loss, motion_loss, texture_map = avatarmodel.optimization(batch_data, first_iter, epoch, kwargs)
                if kwargs.use_appearance:
                    scale_loss = opt.lambda_scale  * scale_loss
                    offset_loss = wdecay_rgl * offset_loss
                    Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                    ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
                else:
                    Ll1, ssim_loss = 0., 0

                # print('loss', ssim_loss.detach().cpu().numpy())
                loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss + motion_loss
            else:
                image, points, pose_loss, offset_loss, = avatarmodel.train_stage2(batch_data, first_iter)

                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                loss = offset_loss + Ll1 + ssim_loss + pose_loss * 10


            if epoch > opt.lpips_start_iter:
                vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                loss = loss + vgg_loss
            
            
            avatarmodel.zero_grad(epoch, kwargs)
            loss.backward(retain_graph=False)
            iter_end.record()

            # print('1', avatarmodel.geo_feature_0[0,0,0,:4].detach().cpu().numpy())
            avatarmodel.step(epoch, kwargs)
            # print('2', avatarmodel.geo_feature_0[0,0,0,:4].detach().cpu().numpy())
            # time_end = time.time()
            # print(' Time %fs' %(time_end - time_start))

            # Progress bar
            if first_iter % 10 == 0:
                # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                pass

            if epoch in [20] and kwargs.use_appearance: #(first_iter-1) % opt.log_iter == 0 and kwargs.save_render and kwargs.use_appearance:
                with torch.no_grad():
                    # save_poitns = points.clone().detach().cpu().numpy()
                    # for i in range(save_poitns.shape[0]):
                    #     pcd = o3d.geometry.PointCloud()
                    #     pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
                    #     o3d.io.write_point_cloud(os.path.join(model.model_path, 'log',"pred_%d.ply" % i) , pcd)

                    avatarmodel.save_rendered(image, gt_image, epoch, batch_data['name'], texture_map)

                    # torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', '{0:05d}_pred'.format(first_iter) + ".png"))
                    # torchvision.utils.save_image(gt_image, os.path.join(model.model_path, 'log', '{0:05d}_gt'.format(first_iter) + ".png"))

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/scale_loss', scale_loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/offset_loss', offset_loss.item(), first_iter)
                # tb_writer.add_scalar('train_loss_patches/aiap_loss', aiap_loss.item(), first_iter)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), first_iter)
                if model.train_stage ==1:
                    tb_writer.add_scalar('train_loss_patches/geo_loss', geo_loss.item(), first_iter)
                else:
                    tb_writer.add_scalar('train_loss_patches/pose_loss', pose_loss.item(), first_iter)
                if epoch > opt.lpips_start_iter:
                    tb_writer.add_scalar('train_loss_patches/vgg_loss', vgg_loss.item(), first_iter)

            t_loss += loss.detach().cpu().numpy()
            t_motion += motion_loss.detach().cpu().numpy()

        # torch.cuda.empty_cache()

        epoch_end = time.time()
        print(' Epoch Time %fs' %(epoch_end - epoch_start))

        if True:
            mpjpe = avatarmodel.eval_smpl(t_loss, t_motion)
            torch.cuda.empty_cache()

        mpjpe = round(float(mpjpe), 5)
        t_motion = round(float(t_motion), 5)

        # if (t_motion < avatarmodel.best_opt_loss or mpjpe < avatarmodel.best_MPJPE) and epoch > avatarmodel.opt_parms.pose_op_start_iter:
        #     print("\n[Epoch %d] Saving Model. MPJPE: %.5f Motion: %.5f" %(epoch, mpjpe, t_motion))
        #     avatarmodel.save(epoch, mpjpe, t_motion)

        avatarmodel.save_all(epoch, mpjpe, t_motion)
        # if epoch > 20:
        #     avatarmodel.vis_smpl(viz=False, save_render=True, save_mesh=True, epoch=epoch, smooth=False)
            
        # early termination
        if epoch > avatarmodel.opt_parms.pose_op_start_iter:
            avatarmodel.best_MPJPE_count += 1
            avatarmodel.best_opt_loss_count += 1

        # early termination
        if (avatarmodel.best_MPJPE_count > 1000000 and avatarmodel.best_opt_loss_count > 10000000):
            if kwargs.save_render:
                avatarmodel.vis_smpl(viz=False, save_render=kwargs.save_render, save_mesh=kwargs.save_mesh, epoch=epoch)
            break

        if epoch in [20] and kwargs.save_render:
            avatarmodel.vis_smpl(viz=False, save_render=kwargs.save_render, save_mesh=kwargs.save_mesh, save_params=kwargs.save_params, epoch=epoch)

        # if epoch % 1 == 0 and epoch > avatarmodel.opt_parms.pose_op_start_iter:
        #     avatarmodel.vis_smpl_frame(viz=False, save_render=kwargs.save_render, save_mesh=kwargs.save_mesh, epoch=epoch)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    np = NetworkParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument("--save_render", action='store_true', default=False)
    parser.add_argument("--save_mesh", action='store_true', default=False)
    parser.add_argument("--save_params", action='store_true', default=False)
    parser.add_argument("--use_prior", action='store_true', default=False)
    parser.add_argument("--use_appearance", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    print(args.source_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(lp.extract(args), np.extract(args), op.extract(args), args)

    print("\nTraining complete.")
