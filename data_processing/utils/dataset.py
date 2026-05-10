'''
 @FileName    : base.py
 @EditTime    : 2022-10-04 15:54:18
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import sys
sys.path.append("./")
import torch.utils.data as data
from utils.FileLoaders import *
import cv2
from alphapose_core.alphapose_module import prepare
from cliff.cliff_module import prepare_cliff
import torch
from yolox.data.data_augment import preproc
from hmr2.hmr2_core import prepare_human4d_data

class dataset(data.Dataset):
    def __init__(self, root, subset, annot, data_type='cliff', eval=False):

        self.root = root
        self.annot = annot
        self.data_type = data_type

        self.seq_id, self.frame_id = [], []
        for s_id, seq in enumerate(self.annot):
            for f_id, frame in enumerate(seq):
                self.seq_id.append(s_id)
                self.frame_id.append(f_id)

        # if eval:
        #     self.pose, self.shape, self.trans = [], [], []
        #     for s_id, seq in enumerate(self.annot):
        #         pose, shape, trans = [], [], []
        #         for f_id, frame in enumerate(seq):
        #             pp, ss, tt = [], [], []
        #             for key in frame.keys():
        #                 if key in ['h_w', 'img_path']:
        #                     continue
        #                 pp.append(frame[key]['pose'])
        #                 ss.append(frame[key]['betas'])
        #                 tt.append(frame[key]['trans'])
        #             pose.append(pp)
        #             shape.append(ss)
        #             trans.append(tt)
        #         self.pose.append(pose)
        #         self.shape.append(shape)
        #         self.trans.append(trans)

        self.len = len(self.seq_id)

    def create_data(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = cv2.imread(os.path.join(self.root, annot['img_path']).replace('\\', '/'))

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data = prepare(img, bboxes)
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data


    def create_scorehmr(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        data = {}

        # hmr2
        BBOX_SHAPE = [192, 256]
        img_size = 256
        IMAGE_MEAN = 255. * np.array([0.485, 0.456, 0.406])
        IMAGE_STD = 255. * np.array([0.229, 0.224, 0.225])

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data_hmr2 = {}
        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img, bboxes, BBOX_SHAPE, img_size,
                                                                                IMAGE_MEAN, IMAGE_STD)
        data_hmr2['img'] = torch.from_numpy(np.array(imgs))
        data_hmr2['personid'] = torch.from_numpy(personid)
        data_hmr2['box_center'] = torch.from_numpy(np.array(box_center))
        data_hmr2['box_size'] = torch.from_numpy(np.array(box_sizes))
        data_hmr2['img_size'] = torch.from_numpy(np.array(img_sizes))
        data_hmr2['img_path'] = img
        data_hmr2["bboxes"] = bboxes

        # cliff
        img = cv2.imread(img)
        intris = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            if annot[person_id]['intri'] is not None:
                intris.append(annot[person_id]['intri'])

        if len(intris) == 0:
            intris = None

        data_cliff = prepare_cliff(img, bboxes, intris=intris)

        # vitpose
        data_vitpose = {}
        data_vitpose["image"] = os.path.join(self.root, annot['img_path'])
        det_results = []
        for box in bboxes:
            bbox = {}
            bbox['bbox'] = np.array(box).reshape(-1)
            det_results.append(bbox)
        data_vitpose["det_results"] = det_results

        data["hmr2"] = data_hmr2
        data["cliff"] = data_cliff
        data["vitpose"] = data_vitpose
        data["seq_id"] = self.seq_id[index]
        data["frame_id"] = self.frame_id[index]
        data["frame_length"] = len(self.annot[self.seq_id[index]])

        return data

    def create_cliff(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = cv2.imread(os.path.join(self.root, annot['img_path']))

        bboxes, intris = [], []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
            if annot[person_id]['intri'] is not None:
                intris.append(annot[person_id]['intri'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        if len(intris) == 0:
            intris = None

        data = prepare_cliff(img, bboxes, intris=intris)
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_trackanything(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        bboxes, intris = [], []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
            if annot[person_id]['intri'] is not None:
                intris.append(annot[person_id]['intri'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        if len(intris) == 0:
            intris = None

        data = {}
        data['bbox'] = bboxes
        data['img_path'] = img
        data['img_name'] = annot['img_path']
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_human4d(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        BBOX_SHAPE = [192, 256]
        img_size = 256
        IMAGE_MEAN = 255. * np.array([0.485, 0.456, 0.406])
        IMAGE_STD =  255. * np.array([0.229, 0.224, 0.225])

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data = {}
        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img, bboxes, BBOX_SHAPE, img_size, IMAGE_MEAN, IMAGE_STD)

        
        data['img'] = torch.from_numpy(np.array(imgs))
        data['personid'] = torch.from_numpy(personid)
        data['box_center'] = torch.from_numpy(np.array(box_center))
        data['box_size'] = torch.from_numpy(np.array(box_sizes))
        data['img_size'] = torch.from_numpy(np.array(img_sizes))

        data['img_path'] = img
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_buddi(self, index):
        data = {}
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data['bbox'] = torch.from_numpy(bboxes)

        data['img_path'] = img
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_bev(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        BBOX_SHAPE = [192, 256]
        img_size = 256
        IMAGE_MEAN = 255. * np.array([0.485, 0.456, 0.406])
        IMAGE_STD =  255. * np.array([0.229, 0.224, 0.225])

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data = {}
        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img, bboxes, BBOX_SHAPE, img_size, IMAGE_MEAN, IMAGE_STD)

        
        data['img'] = torch.from_numpy(np.array(imgs))
        data['personid'] = torch.from_numpy(personid)
        data['box_center'] = torch.from_numpy(np.array(box_center))
        data['box_size'] = torch.from_numpy(np.array(box_sizes))
        data['img_size'] = torch.from_numpy(np.array(img_sizes))

        data['img_path'] = img
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_yolox(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img = img.copy()
            img_info["file_name"] = None

        self.test_size = (800, 1440)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).float()
        if True:
            img = img.half()  # to FP16

        img_info['seq_id'] = self.seq_id[index]
        img_info['frame_id'] = self.frame_id[index]

        return img, img_info

    def create_multihmr(self, index):
        data = {}
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data['bbox'] = torch.from_numpy(bboxes)

        data['img_path'] = img
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data
    
    def create_openpose(self, index):
        data = {}
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])
        del annot['img_path']
        del annot['h_w']
        
        halpe_joints_2d = {}
        for k, v in annot.items():
            halpe_joints_2d[k] = v['halpe_joints_2d']

        data['img_path'] = img
        data['halpe_joints_2d'] = halpe_joints_2d
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def __getitem__(self, index):
        if self.data_type == 'scorehmr':
            return self.create_scorehmr(index)
        elif self.data_type == 'cliff':
            data = self.create_cliff(index)
        elif self.data_type == 'yolox':
            data = self.create_yolox(index)
        elif self.data_type == 'Human4D':
            data = self.create_human4d(index)
        elif self.data_type == 'buddi':
            data = self.create_buddi(index)
        elif self.data_type == 'bev':
            data = self.create_bev(index)
        elif self.data_type == 'multihmr':
            data = self.create_multihmr(index)
        elif self.data_type == 'trackanything':
            data = self.create_trackanything(index)
        elif self.data_type == 'openpose':
            data = self.create_openpose(index)
        else:
            data = self.create_data(index)
        return data

    def __len__(self):
        return self.len



