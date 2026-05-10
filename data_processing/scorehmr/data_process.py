import torch
import numpy as np
import os

from cliff.cliff.utils.geometry import batch_rodrigues
from scorehmr.data_utils import read_images
from utils.FileLoaders import load_pkl

def load_tracking_data(path, img_path):
    imgs = sorted(os.listdir(img_path))
    data = load_pkl(path)

    tracking_data = []
    for i in range(len(imgs)):
        bbox = {key:data[key]['bbox'][i] for key in data.keys()}
        mask = {key:data[key]['segmentation'][i] for key in data.keys()}
        tracking_data.append({'bbox':bbox, 'mask':mask})

    # tracking_data = []
    # for i in range(len(imgs)):
    #     tracking_data.append({key:data[key]['bbox'][i] for key in data.keys()})

    total_number = len(data)
    return tracking_data, total_number

def initialize(img_folder = "demo/demo_test",
                     save_results = False,
                     viz = False,
                     use_samurai = True,
                     crop_size = 224,):
    arg = {
        "img_folder": img_folder,
        "save_results": save_results,
        "viz": viz,
        "use_samurai": use_samurai,
        "crop_size": crop_size,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    hmr2 = {
        "params": [],
        "bbox": [],
        "rendered": [],
        "camparams": [],
        "verts": [],
    }

    cliff = {
        "params": [],
        "rendered": [],
        "camparams": [],
        "verts": [],
    }

    vitpose = {
        "pose": [],
        "img_size": [],
    }

    results = {
        "hmr2": hmr2,
        "cliff": cliff,
        "vitpose": vitpose,
    }

    return arg, results

def initialize_results():
    hmr2 = {
        "params": [],
        "bbox": [],
        "rendered": [],
        "camparams": [],
        "verts": [],
    }
    cliff = {
        "params": [],
        "rendered": [],
        "camparams": [],
        "verts": [],
    }
    vitpose = {
        "pose": [],
        "img_size": [],
    }
    results = {
        "hmr2": hmr2,
        "cliff": cliff,
        "vitpose": vitpose,
    }
    return results


def process_results_hmr2(results, params, bbox, rendered, camparams, verts):
    hmr2 = results["hmr2"]
    hmr2["params"].append(params)
    hmr2["bbox"].append(bbox)
    hmr2["rendered"].append(rendered)
    hmr2["camparams"].append(camparams)
    hmr2["verts"].append(verts)

    return hmr2

def process_reuslts_cliff(results, params, rendered, camparams, verts):
    cliff = results["cliff"]

    cliff["params"].append(params)
    cliff["rendered"].append(rendered)
    cliff["camparams"].append(camparams)
    cliff["verts"].append(verts)

    return cliff

def process_results_vitpose(results, pose, img_size):
    vitpose = results["vitpose"]
    vitpose["pose"].append(pose)
    vitpose["img_size"].append(img_size)

    return vitpose

def data_prepare(arg, results, results_autotrack, imgs, view_num=1):
    hmr2, cliff, vitpose = results["hmr2"], results["cliff"], results["vitpose"]
    device = arg["device"]
    # track_num = len(hmr2["params"][0]["pose"])
    track_num = results_autotrack[1]
    frame_num = len(hmr2["params"])
    data = {}
    data["track_num"] = track_num
    data["frame_num"] = frame_num
    data["view_num"] = view_num
    data["pred_pose"] = torch.zeros((track_num, frame_num, 24, 3, 3), dtype=torch.float32)
    data["pred_betas"] = torch.zeros((track_num, frame_num, 10), dtype=torch.float32)
    data["pred_trans"] = torch.zeros((track_num, frame_num, 3), dtype=torch.float32)
    data["cropped_imgs"] = torch.zeros((track_num, frame_num, 3, arg["crop_size"], arg["crop_size"]), dtype=torch.float32)
    data["joints_2d"] = torch.zeros((track_num, frame_num, 17, 2), dtype=torch.float32)
    data["joints_conf"] = torch.zeros((track_num, frame_num, 17, 1), dtype=torch.float32)

    # hmr2
    for frame_idx, (params, bbox) in enumerate(zip(hmr2["params"], hmr2["bbox"])):
        pose = torch.from_numpy(params["pose"]).reshape(-1, 3).to(device)
        pose = batch_rodrigues(pose).reshape(-1, 24, 3, 3)
        current_track_num = pose.shape[0]
        data["pred_pose"][:current_track_num, frame_idx, :, :, :] = pose
        bbox_center = (bbox[:, :2] + bbox[:, 2:]) / 2
        bbox_scale = bbox[:, 2:] - bbox[:, :2]
        crop = np.stack([
            read_images(os.path.join(arg["img_folder"], imgs[frame_idx]), bbox_c, bbox_s) for bbox_c, bbox_s in zip(bbox_center, bbox_scale)
        ])
        crop = torch.from_numpy(crop).reshape(-1, 3, arg["crop_size"], arg["crop_size"])
        data["cropped_imgs"][:current_track_num, frame_idx, :, :, :] = crop

    # cliff
    for frame_idx, (params, camparams) in enumerate(zip(cliff["params"], cliff["camparams"])):
        trans = torch.from_numpy(params["trans"]).reshape(-1, 3)
        current_track_num = trans.shape[0]
        data["pred_trans"][:current_track_num, frame_idx, :] = trans
        data["focal_length"] = torch.tensor(camparams['intri'][0][0, 0])
        betas = torch.from_numpy(params["betas"]).reshape(-1, 10)
        data["pred_betas"][:current_track_num, frame_idx, :] = betas

    # vitpose
    for frame_idx, (pose, img_size) in enumerate(zip(vitpose["pose"], vitpose["img_size"])):
        for idx, detected in enumerate(pose):
            pred_keypoints = torch.from_numpy(detected["keypoints"][:, :2]).reshape(17, 2)
            keypoints_conf = torch.from_numpy(detected["keypoints"][:, 2:]).reshape(17, 1)
            data["joints_2d"][idx, frame_idx, :, :] = pred_keypoints
            data["joints_conf"][idx, frame_idx, :, :] = keypoints_conf
        data["img_size"] = img_size
    # finally prepare
    pred_pose = data["pred_pose"].to(device).reshape(-1, 24, 3, 3)
    pred_betas = data["pred_betas"].to(device).mean(dim=1).unsqueeze(1).repeat(1, frame_num, 1).reshape(-1, 10)
    pred_trans = data["pred_trans"].to(device).reshape(-1, 3)
    joints_2d = data["joints_2d"].to(device).reshape(-1, 17, 2)
    joints_conf = data["joints_conf"].to(device).reshape(-1, 17, 1)
    cropped_imgs = data["cropped_imgs"].to(device).reshape(-1, 3, arg["crop_size"], arg["crop_size"])
    vis_mask = torch.ones((track_num * frame_num)).to(device)
    img_size = data["img_size"].reshape(1, 2).repeat(track_num * frame_num, 1).to(device)
    camera_center = img_size / 2
    focal_length = data["focal_length"].reshape(1, 1).repeat(track_num * frame_num, 2).to(device)
    data = []
    for view_idx in range(view_num):
        sv_data = {
            "pred_pose": pred_pose,
            "pred_betas": pred_betas,
            "init_cam_t": pred_trans,
            "joints_2d": joints_2d,
            "joints_conf": joints_conf,
            "img": cropped_imgs,
            "vis_mask": vis_mask,
            "img_size": img_size,
            "camera_center": camera_center,
            "focal_length": focal_length,
        }
        data.append(sv_data)

    return data, track_num, frame_num, view_num

def convert_results(params, name="Hi4D"):
    res_pose, res_shape, res_trans = [], [], []
    pose, shape, trans = params["pose"], params["betas"], params["trans"]
    frame_num = pose.shape[1]

    for frame_idx in range(frame_num):
        res_pose.append(pose[:, frame_idx, :])
        res_shape.append(shape[:, frame_idx, :])
        res_trans.append(trans[:, frame_idx, :])

    results = {
        "pose": res_pose,
        "shape": res_shape,
        "trans": res_trans,
    }

    return results

def prepare_scorehmr(results, pre_data, img_path, device, crop_size=224, view_num=1):
    hmr2, cliff, vitpose = results["hmr2"], results["cliff"], results["vitpose"]
    track_num = len(hmr2["params"][0]["pose"])
    frame_num = len(hmr2["params"])
    data = {}
    data["track_num"] = track_num
    data["frame_num"] = frame_num
    data["view_num"] = view_num
    data["pred_pose"] = torch.zeros((track_num, frame_num, 24, 3, 3), dtype=torch.float32)
    data["pred_betas"] = torch.zeros((track_num, frame_num, 10), dtype=torch.float32)
    data["pred_trans"] = torch.zeros((track_num, frame_num, 3), dtype=torch.float32)
    data["cropped_imgs"] = torch.zeros((track_num, frame_num, 3, crop_size, crop_size), dtype=torch.float32)
    data["joints_2d"] = torch.zeros((track_num, frame_num, 17, 2), dtype=torch.float32)
    data["joints_conf"] = torch.zeros((track_num, frame_num, 17, 1), dtype=torch.float32)

    # hmr2
    for frame_idx, (params, pd) in enumerate(zip(hmr2["params"], pre_data)):
        pose = torch.from_numpy(params["pose"]).reshape(-1, 3).to(device)
        pose = batch_rodrigues(pose).reshape(-1, 24, 3, 3)
        data["pred_pose"][:, frame_idx, :, :, :] = pose
        betas = torch.from_numpy(params["betas"]).reshape(-1, 10)
        data["pred_betas"][:, frame_idx, :] = betas
        bbox = pd["hmr2"]["bboxes"].cpu().numpy().reshape(-1, 4)
        bbox_center = (bbox[:, :2] + bbox[:, 2:]) / 2
        bbox_scale = bbox[:, 2:] - bbox[:, :2]
        crop = np.stack([
            read_images(img_path, bbox_c, bbox_s) for bbox_c, bbox_s in zip(bbox_center, bbox_scale)
        ])
        crop = torch.from_numpy(crop).reshape(-1, 3, crop_size, crop_size)
        data["cropped_imgs"][:, frame_idx, :, :, :] = crop

    # cliff
    for frame_idx, (params, pd) in enumerate(zip(cliff["params"], pre_data)):
        trans = params["pred_cam_t"]
        data["pred_trans"][:, frame_idx, :] = trans
        data["focal_length"] = torch.tensor(int(pd["cliff"]["focal_length"][0]))

    # vitpose
    for frame_idx, (pose, pd) in enumerate(zip(vitpose["pose"], pre_data)):
        for idx, detected in enumerate(pose):
            pred_keypoints = torch.from_numpy(detected["keypoints"][:, :2]).reshape(17, 2)
            keypoints_conf = torch.from_numpy(detected["keypoints"][:, 2:]).reshape(17, 1)
            data["joints_2d"][idx, frame_idx, :, :] = pred_keypoints
            data["joints_conf"][idx, frame_idx, :, :] = keypoints_conf
        data["img_size"] = pd["hmr2"]["img_size"][0]

    # finally prepare
    pred_pose = data["pred_pose"].to(device).reshape(-1, 24, 3, 3)
    pred_betas = data["pred_betas"].to(device).reshape(-1, 10)
    pred_trans = data["pred_trans"].to(device).reshape(-1, 3)
    joints_2d = data["joints_2d"].to(device).reshape(-1, 17, 2)
    joints_conf = data["joints_conf"].to(device).reshape(-1, 17, 1)
    cropped_imgs = data["cropped_imgs"].to(device).reshape(-1, 3, crop_size, crop_size)
    vis_mask = torch.ones((track_num * frame_num)).to(device)
    img_size = data["img_size"].reshape(1, 2).repeat(track_num * frame_num, 1).to(device)
    camera_center = img_size / 2
    focal_length = data["focal_length"].reshape(1, 1).repeat(track_num * frame_num, 2).to(device)
    data = []
    for view_idx in range(view_num):
        sv_data = {
            "pred_pose": pred_pose,
            "pred_betas": pred_betas,
            "init_cam_t": pred_trans,
            "joints_2d": joints_2d,
            "joints_conf": joints_conf,
            "img": cropped_imgs,
            "vis_mask": vis_mask,
            "img_size": img_size,
            "camera_center": camera_center,
            "focal_length": focal_length,
        }
        data.append(sv_data)

    return data, track_num, frame_num, view_num