'''
 @FileName    : vitpose.py
 @EditTime    : 2023-02-03 16:18:05
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import sys
sys.path.append("./vitpose")
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from utils.module_utils import draw_keyp, vis_img

class ViTPose_Predictor(object):
    def __init__(
        self,
        vitpose_type,
        pose_type,
        checkpoint_type,
        thres,
        decoder=None,
        device='cuda:0',
        fp16=True
    ):
        assert vitpose_type in ['vitpose', 'vitpose_plus']
        
        if vitpose_type == 'vitpose':
            assert pose_type == 'coco'
            assert checkpoint_type in ['base', 'large', 'base*', 'large*', 'huge*']
        
            ck_paths = {
                'base': 'pretrained/vitpose_data/vitpose_base_coco_aic_mpii.pth',
                'large': 'pretrained/vitpose_data/vitpose_large_coco_aic_mpii.pth',
                'base*': 'pretrained/vitpose_data/vitpose-b-multi-coco.pth',
                'large*': 'pretrained/vitpose_data/vitpose-l-multi-coco.pth',
                'huge*': 'pretrained/vitpose_data/vitpose-h-multi-coco.pth'
            }
            
            pose_checkpoint = ck_paths[checkpoint_type]
            checkpoint_type = checkpoint_type.replace('*', '')
            pose_config = f'vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/{pose_type}/ViTPose_{checkpoint_type}_{pose_type}_256x192.py'
        else:
            assert checkpoint_type in ['small', 'base', 'large', 'huge']
            assert pose_type in ['coco', 'wholebody']
            pose_checkpoint = f'pretrained/vitpose_data/vitpose+_{checkpoint_type}/{pose_type}.pth'
            if pose_type == 'wholebody':
                pose_config = f'vitpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_{checkpoint_type}_wholebody_256x192.py'
            else:
                assert checkpoint_type != 'small' 'vitpose+ small checkpoint not support coco format'
                pose_config = f'vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_{checkpoint_type}_coco_256x192.py'
                
        self.pose_type = pose_type
        self.pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device)

    def inference(self, data):
        dataset_name = self.pose_model.cfg.data['test']['type']
        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            data["image"],
            data["det_results"],
            bbox_thr=None,
            format='xyxy',
            dataset=dataset_name
        )

        return pose_results

    def predict(self, image, bboxes, masks=None):

        det_results = []

        for box in bboxes:
            bbox = {}
            bbox['bbox'] = np.array(box).reshape(-1)
            det_results.append(bbox)
  
        # test a single image, with a list of bboxes
        dataset_name = self.pose_model.cfg.data['test']['type']
        pose_results, _ = inference_top_down_pose_model(
                    self.pose_model,
                    image,
                    det_results,
                    bbox_thr=None,
                    format='xyxy',
                    masks=masks,
                    dataset=dataset_name)

        return pose_results


    def wholebody2halpe(self, poses):

        for person in poses:
            halpe = np.zeros((26,3))
            coco_whole_joints = person['keypoints']

            halpe[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21,22,23,24,25]] = coco_whole_joints[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,18,21,19,22]]
            # hip
            halpe[19] = (coco_whole_joints[11] + coco_whole_joints[12])/2
            # neck
            halpe[18] = (coco_whole_joints[5] + coco_whole_joints[6])/2

            person['keypoints'] = halpe

        return poses

    def visualize(self, img, poses, format='halpe', viz=False):
        
        img = img.copy()

        if self.pose_type == 'wholebody' and format == 'halpe':
            
            poses = self.wholebody2halpe(poses)
            for person in poses:
                img = draw_keyp(img, person['keypoints'], format=format)
        
        elif self.pose_type == 'coco':
            for person in poses:
                img = draw_keyp(img, person['keypoints'], format='coco17')

        if viz:
            vis_img('image', img)

        return img