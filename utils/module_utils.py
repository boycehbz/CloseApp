'''
 @FileName    : module_utils.py
 @EditTime    : 2022-09-27 14:38:28
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import random
import torch
import cv2
import math
import os
import uuid
import time
import yaml
import cv2

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int32)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img

def overlay_mask_on_image(image, mask, alpha=0.5):
    
    result_image = image.copy()

    for m in mask:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        color = (b, g, r)
        class_mask = (m == 1).astype(np.uint8)
        result_image[class_mask == 1] = (
            1 - alpha) * result_image[class_mask == 1] + alpha * np.array(color)

    return result_image

def prepare_output_and_logger(args, net, opt, checkpoint):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    args.model_path = os.path.join(args.model_path, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    os.makedirs(os.path.join(args.model_path, 'log'), exist_ok = True)
    # with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
    #     cfg_log_f.write(str(Namespace(**vars(args))))

    # Store the arguments for the current experiment
    conf_fn = os.path.join(args.model_path, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)
        yaml.dump(net, conf_file)
        yaml.dump(opt, conf_file)
        yaml.dump(checkpoint, conf_file)

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

def vis_img(name, im):
    ratiox = 1200/int(im.shape[0])
    ratioy = 1200/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    cv2.waitKey()

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_camera_para(file):
    """"
    load camera parameters
    """
    campose = []
    intra = []
    campose_ = []
    intra_ = []
    f = open(file,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if len(words) == 3:
            intra_.append([float(words[0]),float(words[1]),float(words[2])])
        elif len(words) == 4:
            campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
        else:
            pass

    index = 0
    intra_t = []
    for i in intra_:
        index+=1
        intra_t.append(i)
        if index == 3:
            index = 0
            intra.append(intra_t)
            intra_t = []

    index = 0
    campose_t = []
    for i in campose_:
        index+=1
        campose_t.append(i)
        if index == 3:
            index = 0
            campose_t.append([0.,0.,0.,1.])
            campose.append(campose_t)
            campose_t = []
    
    return np.array(campose), np.array(intra)

def save_camparam(path, intris, extris):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    f = open(path, 'w')
    for ind, (intri, extri) in enumerate(zip(intris, extris)):
        f.write(str(ind)+'\n')
        for i in intri:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
        f.write('0 0 \n')
        for i in extri[:3]:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
        f.write('\n')
    f.close()

def convert_color(gray):
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=1),cv2.COLORMAP_JET)
    return im_color

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

