from pathlib import Path
import sys
import os

DATA_PROCESSING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_PROCESSING_ROOT.parent
SAM2_ROOT = DATA_PROCESSING_ROOT / "samurai" / "sam2"

for path in (PROJECT_ROOT, SAM2_ROOT, DATA_PROCESSING_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
# insert the paths at the beginning of sys.path
sys.path.insert(0, str(DATA_PROCESSING_ROOT))
sys.path.insert(1, str(SAM2_ROOT))
sys.path.insert(2, str(PROJECT_ROOT))

os.chdir(DATA_PROCESSING_ROOT)
os.environ["PYOPENGL_PLATFORM"] = "egl" #osmesa egl
import cv2
import numpy as np
import torch
from utils.projection import joint_projection
from utils.FileLoaders import *
from hmr2.hmr2_core import Human4D_Predictor
from utils.smpl_torch_batch import SMPLModel
from utils.module_utils import estimate_translation_np, vis_img, copy, annToMask
from utils.renderer_pyrd import Renderer
from samurai.samurai_core import SAMURAI_core
from yolox.yolox import Predictor
from scorehmr.data_process import *
from tqdm import tqdm
from cliff.cliff_core import CLIFF_Predictor
from vitpose.vitpose_core import ViTPose_Predictor
from scorehmr.scorehmr import ScoreHMR

viz = False
root = 'data/images'
out = 'output/preprocess_data'

seqs = sorted(os.listdir(root))

smpl = SMPLModel(model_path='smpl/smpl/SMPL_NEUTRAL.pkl')
halpe_regressor = np.load('smpl/J_regressor_halpe.npy')

model_dir = 'pretrained/yolox_data/bytetrack_x_mot17.pth.tar'
thres = 0.23
yolox_predictor = Predictor(model_dir, thres)

model_dir = 'pretrained/Human4D_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
human4d_predictor = Human4D_Predictor(model_dir)

sam_model_dir = 'pretrained/samurai_data/sam2.1_hiera_base_plus.pt'
samurai = SAMURAI_core(sam_model_dir, yolox_predictor)

# CLIFF
model_dir = R'pretrained/cliff_data/cliff_trained.pt'
cliff_predictor = CLIFF_Predictor(model_dir, 'hr48')

# vitpose
vitpose_type = 'vitpose_plus' # vitpose or vitpose_plus
pose_type = 'coco' # coco or wholebody
checkpoint_type = 'huge'
vit_thres = 0.1
vit_predictor = ViTPose_Predictor(vitpose_type, pose_type, checkpoint_type, vit_thres)


for s_id, seq in enumerate(seqs):

    seq_path = os.path.join(root, seq)
    output = os.path.join(out, seq, 'train')

    img_folder = seq_path

    arg, results = initialize(img_folder=img_folder,
                            save_results=False,
                            viz=viz,
                            use_samurai=True,
                            crop_size = 224,)
    
    imgs = sorted(os.listdir(img_folder))

    try:
        contact = load_pkl(os.path.join('/media/buzhenhuang/Data/WildCHI_test/contact_new', seq + '.pkl'))
        # contact['label'] = contact['label'][:len(imgs)]
        # contact['label'] = contact['label'][27:]
    except:
        contact = None
        
    try:
        det = load_pkl(os.path.join('/media/buzhenhuang/Data/WildCHI_source/tracking_results', seq, 'results.pkl'))
        bboxes = [det[key]['bbox'][0] for key in det.keys()]
    except:
        img_path = os.path.join(arg["img_folder"], imgs[0])
        bbox, _ = yolox_predictor.predict(img_path)
        bboxes = bbox['bbox'][:2]

    results_autotrack = samurai.inference(arg["img_folder"], bboxes=bboxes, viz=False)

    param_data = {'pred_bbox':[], 'pred_pose':[], 'pred_trans':[], 'pred_shape':[], 'gender':['neutral']*2, 'features':[], 'init_pose':[], 'img_w':[], 'img_h':[], 'focal_length':[], 'center':[], 'scale':[], 'keypoints':[], 'pred_keypoints':[], 'valid':[], }

    for frame_idx, (img_name, tracking_data) in tqdm(enumerate(zip(imgs, results_autotrack[0])), total=len(imgs)):
        # hmr2
        img_path = os.path.join(arg["img_folder"], img_name)
        img = cv2.imread(img_path)
        img_size = torch.tensor([img.shape[1], img.shape[0]])
        # t_data = {}
        # t_data['bbox'] = tracking_data
        bbox = np.array([tracking_data['bbox'][k] for k in tracking_data['bbox'].keys()], dtype=np.float32)
        total_person = len(bbox)

        param_data['pred_bbox'].append(bbox)

        if 'mask' in tracking_data.keys():
            mask = []
            for k in tracking_data['mask'].keys():
                if len(tracking_data['mask'][k]) > 0:
                    m = annToMask(tracking_data['mask'][k], img.shape[0], img.shape[1])
                else:
                    m = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
                mask.append(m)

                mask_out_path = os.path.join(output, 'masks', k, img_name.replace('jpg', 'png'))
                os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                cv2.imwrite(mask_out_path, m*255)

        else:
            mask = None

        human4d_params, human4d_rendered, human4d_camparams, human4d_verts = human4d_predictor.get_closeint_features(img_path, bbox, viz=False) #human4d_predictor.predict(img_path, bbox, viz=False)
        results["hmr2"] = process_results_hmr2(results, human4d_params, bbox, human4d_rendered, human4d_camparams, human4d_verts)

        # CLIFF
        params, rendered, camparams, verts = cliff_predictor.predict(img_path, bbox, viz=False)
        results["cliff"] = process_reuslts_cliff(results, params, rendered, camparams, verts)

        # vitpose
        pose = vit_predictor.predict(img, bbox, mask)
        result_img = vit_predictor.visualize(img, pose, viz=False)
        results["vitpose"] = process_results_vitpose(results, pose, img_size)

        # joints_2d = np.array([p['keypoints'] for p in pose])
        # temp = np.zeros((joints_2d.shape[0], 9, 3), dtype=np.float32)
        # joints_2d = np.concatenate((joints_2d, temp), axis=1)

        img_h, img_w = img.shape[:2]
        focal = (img_h**2 + img_w**2)**0.5

        joints_3d = halpe_regressor @ human4d_verts
        h4d_intri = human4d_camparams['intri'][0]
        extri = human4d_camparams['extri'][0]
        joints_2d, _ = joint_projection(joints_3d.reshape(-1, 3), extri, h4d_intri, img, viz=False)
        
        joints_2d = joints_2d.reshape(total_person, -1, 2)
        conf = np.ones((joints_2d.shape[0], joints_2d.shape[1], 1))
        joints_2d = np.concatenate((joints_2d, conf), axis=2)

        param_data['features'].append(human4d_params['features'])
        param_data['init_pose'].append(human4d_params['pose'])
        param_data['img_w'].append([img_w]*total_person)
        param_data['img_h'].append([img_h]*total_person)
        param_data['focal_length'].append([focal]*total_person)
        param_data['center'].append(human4d_params['centers'])
        param_data['scale'].append(human4d_params['scales'])
        param_data['keypoints'].append(joints_2d)
        param_data['pred_keypoints'].append(joints_2d)
        param_data['valid'].append([1]*total_person)

        img_out_path = os.path.join(output, 'images', img_name)
        os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
        cv2.imwrite(img_out_path, img)

    # Data prepare
    data, track_num, frame_num, view_num = data_prepare(arg, results, results_autotrack, imgs, view_num=1)
    data[0]['contact'] = contact

    # scorehmr
    scorehmr = ScoreHMR(device=arg["device"])
    params, vertices = scorehmr.iterate(data,
                                        track_num=track_num,
                                        frame_num=frame_num,
                                        image_folder=arg["img_folder"],
                                        viz=False,
                                        save_results=False)

    assert len(param_data['pred_bbox']) == params['pose'].shape[1]

    for idx in range(params['pose'].shape[1]):
        param_data['pred_pose'].append(params['pose'][:,idx])
        param_data['pred_trans'].append(params['trans'][:,idx])
        param_data['pred_shape'].append(params['betas'][:,idx])

    for key in param_data.keys():
        if key not in ['gender']:
            param_data[key] = torch.from_numpy(np.array(param_data[key])).float()

    param_data['contact'] = contact

    torch.save(param_data, os.path.join(output, 'smpl_parms.pth'))

    intrinsic = np.eye(3, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    img_h, img_w = img.shape[:2]
    intrinsic[0][0] = (img_h**2 + img_w**2) ** 0.5
    intrinsic[1][1] = (img_h**2 + img_w**2) ** 0.5
    intrinsic[0][2] = img_w / 2.
    intrinsic[1][2] = img_h / 2.
    save_npz(os.path.join(output, 'cam_parms.npz'), {'intrinsic':intrinsic, 'extrinsic':extrinsic})

    files = os.listdir('data/uv')
    for f in files:
        src = os.path.join('data/uv', f)
        dst = os.path.join(output, f)
        copy(src, dst)
