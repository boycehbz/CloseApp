import cv2
import numpy as np
from scorehmr.datasets.utils import generate_image_patch, convert_cvimg_to_tensor

DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])


def read_images(img_path, box_center, box_scale, crop_res=224):
    if box_center.sum() > 0:
        # read images -- vis_mask 1 or 0
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError(f"Fail to read {img_path}")
        img_height, img_width, img_channels = cvimg.shape

        img_patch_cv, _ = generate_image_patch(
            img=cvimg,
            c_x=box_center[0],
            c_y=box_center[1],
            bb_width=box_scale.max(),
            bb_height=box_scale.max(),
            patch_width=crop_res,
            patch_height=crop_res,
            do_flip=False,
            scale=1.0,
            rot=0.0,
            load_image=True,
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        for n_c in range(min(img_channels, 3)):
            img_patch[n_c, :, :] = (
                img_patch[n_c, :, :] - DEFAULT_MEAN[n_c]
            ) / DEFAULT_STD[n_c]
    else:
        # corresponds to vis_mask=-1 (track out of scene) and will not be used
        img_patch = np.zeros((3, crop_res, crop_res), dtype=np.float32)
    return img_patch