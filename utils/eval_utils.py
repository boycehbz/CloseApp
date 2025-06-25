import torch
import torch.nn as nn
import numpy as np
import cv2
import os

def rearrange_joints2frame(pred_joints, file_names):
    pred_poses = np.concatenate(pred_joints)

    pred_ps, pred_ss, pred_ts, pred_2ds, names = [], [], [], [], []
    name_last = None
    pred_p, pred_s, pred_t, pred_2 = [], [], [], []
    for name, pred_pose in zip(file_names, pred_poses):
        if name != name_last:
            if name_last is not None:
                names.append(name_last)
                pred_ps.append(np.array(pred_p))
            name_last = name
            pred_p, pred_s, pred_t, pred_2 = [], [], [], []
        pred_p.append(pred_pose)

    names.append(name_last)
    pred_ps.append(np.array(pred_p))
    return pred_ps, names

def rearrange2frame(pred_poses, pred_shapes, pred_transs, pred2ds, file_names):
    pred_poses = np.concatenate(pred_poses)
    pred_shapes = np.concatenate(pred_shapes)
    pred_transs = np.concatenate(pred_transs)
    pred2ds = np.concatenate(pred2ds)

    pred_ps, pred_ss, pred_ts, pred_2ds, names = [], [], [], [], []
    name_last = None
    pred_p, pred_s, pred_t, pred_2 = [], [], [], []
    for name, pred_pose, pred_shape, pred_trans, pred_2d in zip(file_names, pred_poses, pred_shapes, pred_transs, pred2ds):
        if name != name_last:
            if name_last is not None:
                names.append(name_last)
                pred_ps.append(np.array(pred_p))
                pred_ss.append(np.array(pred_s))
                pred_ts.append(np.array(pred_t))
                pred_2ds.append(np.array(pred_2))
            name_last = name
            pred_p, pred_s, pred_t, pred_2 = [], [], [], []
        pred_p.append(pred_pose)
        pred_s.append(pred_shape)
        pred_t.append(pred_trans)
        pred_2.append(pred_2d)
    names.append(name_last)
    pred_ps.append(np.array(pred_p))
    pred_ss.append(np.array(pred_s))
    pred_ts.append(np.array(pred_t))
    pred_2ds.append(np.array(pred_2))
    return pred_ps, pred_ss, pred_ts, pred_2ds, names

def rearrange2seq(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids, file_names, is_seq=True):
    gt_ps, gt_ss, pred_ps, pred_ss, preds, vals = [], [], [], [], [], []
    name_last = None
    gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
    for name, pred_pose, pred_shape, gt_pose, gt_shape, gt_joint, valid in zip(file_names, pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids):
        name = os.path.dirname(name)
        if name != name_last and is_seq:
            if name_last is not None:
                gt_ps.append(np.array(gt_p))
                gt_ss.append(np.array(gt_s))
                pred_ps.append(np.array(pred_p))
                pred_ss.append(np.array(pred_s))
                preds.append(np.array(pred))
                vals.append(np.array(val))
            name_last = name
            gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
        elif len(gt_p) >= 2000 and not is_seq:
            gt_ps.append(np.array(gt_p))
            gt_ss.append(np.array(gt_s))
            pred_ps.append(np.array(pred_p))
            pred_ss.append(np.array(pred_s))
            preds.append(np.array(pred))
            vals.append(np.array(val))
            gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
        gt_p.append(gt_pose)
        gt_s.append(gt_shape)
        pred_p.append(pred_pose)
        pred_s.append(pred_shape)
        pred.append(gt_joint)
        val.append(valid)
    gt_ps.append(np.array(gt_p))
    gt_ss.append(np.array(gt_s))
    pred_ps.append(np.array(pred_p))
    pred_ss.append(np.array(pred_s))
    preds.append(np.array(pred))
    vals.append(np.array(val))
    return gt_ps, gt_ss, pred_ps, pred_ss, preds, vals

def cal_ordinal(pd1, pd2, gt1, gt2, thres):
    if (gt1 - gt2) * (pd1 - pd2) > 0:
        ordi = 1
    else:
        if abs(gt1 - gt2) < thres and abs(pd1 - pd2) < thres:
            ordi = 0
        else:
            ordi = -1
    return ordi


def PCOD(pred_rt_Z, gt_rt_Z, thres=0.3):
    '''
    input:  pred_rt_Z:predicted depth of roots   N*1
            gt_rt_Z:gt depth of roots            N*1
            thres:threshold of distance between two people, if lower than it we think it is correct
            
            N*1: N is the number of people in one frame, 1 is the depth of a root joint
    
    output: PCOD 'percentage of correct ordinal depth (PCOD) relations between people'
    '''
    total_ordinal = 0
    correct_ordinal = 0

    if len(pred_rt_Z) >= 2: 
        for irt in range(len(pred_rt_Z) - 1):
            for irt_hd in range(irt+1, len(pred_rt_Z)):
                ordi = cal_ordinal(pred_rt_Z[irt_hd], pred_rt_Z[irt], gt_rt_Z[irt_hd], gt_rt_Z[irt], thres)

                if ordi >= 0:
                    correct_ordinal = correct_ordinal + 1
                
                total_ordinal = total_ordinal + 1
    
    return correct_ordinal / total_ordinal



def dist(p1, p2, th):

    """
    3D Point Distance
    :param p1: predicted point
    :param p2: GT point
    :param th: max acceptable distance
    :return: euclidean distance between the positions of the two joints
    """
    if p1[0] != p2[0]:
        return np.nan
    d = np.linalg.norm(np.array(p1) - np.array(p2))
    return d if d <= th else np.nan


def non_minima_suppression(x: np.ndarray) -> np.ndarray:
    """
    :return: non-minima suppressed version of the input array;
    supressed values become np.nan
    """
    min = np.nanmin(x)
    x[x != min] = np.nan
    if len(x[x == min]) > 1:
        ok = True
        for i in range(len(x)):
            if x[i] == min and ok:  
                ok = False
            else:
                x[i] = np.nan
    return x


def not_nan_count(x: np.ndarray) -> int:
    """
    :return: number of not np.nan elements of the array
    """
    return len(x[~np.isnan(x)])


def joint_det_metrics(points_pred, points_true, th=7.0):

    """
    Joint Detection Metrics
    :param points_pred: list of predicted points
    :param points_true: list of GT points
    :param th: distance threshold; all distances > th will be considered 'np.nan'.
    :return: a dictionary of metrics, 'met', related to joint detection;
             the the available metrics are:
             (1) met['tp'] = number of True Positives
             (2) met['fn'] = number of False Negatives
             (3) met['fp'] = number of False Positives
             (4) met['pr'] = PRecision
             (5) met['re'] = REcall
             (6) met['f1'] = F1-score
    """
    # create distance matrix
    # the number of rows of the matrix corresponds to the number of GT joints
    # the number of columns of the matrix corresponds to the number of predicted joints
    # mat[i,j] contains the njd-distance between joints_true[i] and joints_pred[j]

    if len(points_pred) > 0 and len(points_true) > 0:
        mat = []
        for p_true in points_true:
            row = np.array([dist(p_pred, p_true, th=th) for p_pred in points_pred])
            mat.append(row)
        mat = np.array(mat)
        mat = np.apply_along_axis(non_minima_suppression, 1, mat)
        mat = np.apply_along_axis(non_minima_suppression, 0, mat)

        # calculate joint detection metrics
        nr = np.apply_along_axis(not_nan_count, 1, mat)
        tp = len(nr[nr != 0])  # number of True Positives
        fn = len(nr[nr == 0])  # number of False Negatives
        fp = len(points_pred) - tp  # number of False Positives
        pr = tp / (tp + fp)  # PRecision
        re = tp / (tp + fn)  # REcall
        f1 = 2 * tp / (2 * tp + fn + fp)  # F1-score

    elif len(points_pred) == 0 and len(points_true) == 0:
        tp = 0  # number of True Positives
        fn = 0  # number of False Negatives
        fp = 0
        pr = 1.0
        re = 1.0
        f1 = 1.0
    elif len(points_pred) == 0:
        tp = 0  # number of True Positives
        fn = len(points_true)  # number of False Negatives
        fp = 0
        pr = 0.0  # PRecision
        re = 0.0  # REcall
        f1 = 0.0  # F1-score
    else:
        tp = 0
        fn = 0
        fp = len(points_pred)
        pr = 0.0  # PRecision
        re = 0.0  # REcall
        f1 = 0.0  # F1-score

    # build the metrics dictionary
    metrics = {
        'tp': tp, 'fn': fn, 'fp': fp,
        'pr': pr, 're': re, 'f1': f1,
    }

    return metrics

def f1_score(coords3d_pred, coords3d_true):
    '''
    input:  coords3d_pred: predicted 3D poses   N*4
            coords3d_true: gt 3D poses          N*4
            N*4: N is the number of all the joints in one frame; 4 is the joint index and 3D position

    output: metrics_dict: a dict of PRecision\REcall\F1-score with different threshold 
            metrics_dict = ['pr': , 're': , 'f1': ]
    '''
    # metrics thresholds
    THS = [0.4, 0.8, 1.2]
    
    metrics_dict = {}
    for th in THS:
        for key in ['pr', 're', 'f1']:
            metrics_dict[f'{key}@{th}'] = []  

    for th in THS:
            __m = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=th)
            for key in ['pr', 're', 'f1']:
                metrics_dict[f'{key}@{th}'] = __m[key]

    return metrics_dict

class HumanEval(nn.Module):
    def __init__(self, name, generator=None, smpl=None, dtype=torch.float32, **kwargs):
        super(HumanEval, self).__init__()
        self.generator = generator
        self.smpl = smpl
        if dtype == 'float32':
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64
        self.name = name
        self.dataset_scale = self.dataset_mapping(self.name)
        self.J_regressor_H36 = np.load('data/smpl/J_regressor_h36m.npy').astype(np.float32)
        self.J_regressor_LSP = np.load('data/smpl/J_regressor_lsp.npy').astype(np.float32)
        self.J_regressor_Halpe = np.load('data/smpl/J_regressor_halpe.npy').astype(np.float32)
        self.J_regressor_SMPL = self.smpl.J_regressor.clone().cpu().detach().numpy()

        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

        self.eval_handler_mapper = dict(
            Hi4D=self.LSPEvalHandler,
            JTA=self.LSPEvalHandler,
            AGORA=self.SMPLEvalHandler,
            OcMotion=self.LSPEvalHandler,
            Human36M_MOSH=self.LSPEvalHandler,
            VCL_3DOH50K=self.LSPEvalHandler,
            VCLMP=self.LSPEvalHandler,
            h36m_synthetic_protocol2=self.LSPEvalHandler,
            h36m_valid_protocol1=self.LSPEvalHandler,
            h36m_valid_protocol2=self.LSPEvalHandler,
            MPI3DPW=self.LSPEvalHandler,
            MPI3DPW_singleperson=self.LSPEvalHandler,
            MPI3DPWOC=self.LSPEvalHandler,
            Panoptic_haggling1=self.PanopticEvalHandler,
            Panoptic_mafia2=self.PanopticEvalHandler,
            Panoptic_pizza1=self.PanopticEvalHandler,
            Panoptic_ultimatum1=self.PanopticEvalHandler,
            Panoptic_Eval=self.PanopticEvalHandler,
            MuPoTS_origin=self.MuPoTSEvalHandler,
            MPI3DHP=self.MuPoTSEvalHandler,
        )

    def init_lists(self):
        self.vertex_error, self.error, self.error_pa, self.abs_pck, self.pck, self.accel = [], [], [], [], [], []

    def report(self):
        vertex_error = np.mean(np.array(self.vertex_error))
        error = np.mean(np.array(self.error))
        error_pa = np.mean(np.array(self.error_pa))
        abs_pck = np.mean(np.array(self.abs_pck))
        pck = np.mean(np.array(self.pck))
        accel = np.mean(np.array(self.accel))
        return vertex_error, error, error_pa, abs_pck, pck, accel

    def dataset_mapping(self, name):
        if name == 'VCLMP':
            return 105
        if name == 'VCL_3DOH50K':
            return 7
        else:
            return 1

    def estimate_translation_from_intri(self, S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
        num_joints = S.shape[0]
        # focal length
        f = np.array([fx, fy])
        # optical center
    # center = np.array([img_size/2., img_size/2.])
        center = np.array([cx, cy])
        # transformations
        Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
        XY = np.reshape(S[:,0:2],-1)
        O = np.tile(center,num_joints)
        F = np.tile(f,num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

        # least squares
        Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
        c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W,Q)
        c = np.dot(W,c)

        # square matrix
        A = np.dot(Q.T,Q)
        b = np.dot(Q.T,c)

        # test
        A += np.eye(A.shape[0]) * 1e-6

        # solution
        trans = np.linalg.solve(A, b)
        return trans

    def cal_trans(self, J3ds, J2ds, intris):
        trans = np.zeros((J3ds.shape[0], 3))
        for i, (J3d, J2d, intri) in enumerate(zip(J3ds, J2ds, intris)):
            fx = intri[0][0]
            fy = intri[1][1]
            cx = intri[0][2]
            cy = intri[1][2]
            j_conf = J2d[:,2] 
            trans[i] = self.estimate_translation_from_intri(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
        return trans

    def get_abs_meshes(self, pre_meshes, joints_2ds, intri):
        lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        pre_meshes = ((pre_meshes + 0.5) * 2. * self.dataset_scale)
        # get predicted 3D joints and estimate translation
        joints = np.matmul(self.J_regressor_LSP, pre_meshes)
        # we use 12 joints to calculate translation
        transl = self.cal_trans(joints[:,lsp14_to_lsp13], joints_2ds, intri)

        abs_mesh = pre_meshes + transl[:,np.newaxis,:]
        return abs_mesh

    def compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale*(R.dot(mu1))

        # 7. Error:
        S1_hat = scale*R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

    def align_by_pelvis_batch(self, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['h36m']:
            pelvis_id = 0
            pelvis = joints[:,pelvis_id, :]
        elif format in ['smpl']:
            left_id = 1
            right_id = 2
            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[:,pelvis_id, :]
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=1), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=1)

    def align_by_pelvis(self, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=0)

    def align_mesh_by_pelvis_batch(self, mesh, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2
            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[:,pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[:,pelvis_id, :]
        if get_pelvis:
            return mesh - np.expand_dims(pelvis, axis=1), pelvis
        else:
            return mesh - np.expand_dims(pelvis, axis=1)

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        S1 = torch.from_numpy(S1).float()
        S2 = torch.from_numpy(S2).float()
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat.numpy()

    def compute_errors(self, gt3ds, preds, valids, format='lsp', confs=None):
        if confs is None:
            confs = np.ones((gt3ds.shape[:2]))

        abs_errors = (np.mean(np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        joint_error = np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs) * valids[:,np.newaxis]
        abs_pck = np.mean(joint_error < 150, axis=1).tolist()

        gt3ds = self.align_by_pelvis_batch(gt3ds, format=format)
        preds = self.align_by_pelvis_batch(preds, format=format)

        errors = (np.mean(np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        joint_error = np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs) * valids[:,np.newaxis]
        pck = np.mean(joint_error < 150, axis=1).tolist()

        accel_err = np.zeros((len(gt3ds,)))
        accel_err[1:-1] = self.compute_error_accel(joints_pred=preds, joints_gt=gt3ds)
        accel = (accel_err * valids).tolist()

        preds_sym = self.batch_compute_similarity_transform(preds, gt3ds)
        errors_pa = (np.mean(np.sqrt(np.sum((gt3ds - preds_sym) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        # abs_errors, errors, errors_pa, abs_pck, pck, gt_joints, pred_joints = [], [], [], [], [], [], []
        # for i, (gt3d, pred, conf) in enumerate(zip(gt3ds, preds, confs)):
        #     gt3d = gt3d.reshape(-1, 3)

        #     # Get abs error.
        #     joint_error = np.sqrt(np.sum((gt3d - pred)**2, axis=1)) * conf
        #     abs_errors.append(np.mean(joint_error))

        #     # Get abs pck.
        #     abs_pck.append(np.mean(joint_error < 150) * 100)

        #     # Root align.
        #     gt3d = self.align_by_pelvis(gt3d, format=format)
        #     pred3d = self.align_by_pelvis(pred, format=format)

        #     gt_joints.append(gt3d)
        #     pred_joints.append(pred3d)

        #     joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1)) * conf
        #     errors.append(np.mean(joint_error))

        #     # Get pck
        #     pck.append(np.mean(joint_error < 150) * 100)

        #     # Get PA error.
        #     pred3d_sym = self.compute_similarity_transform(pred3d, gt3d)
        #     pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1)) * conf
        #     errors_pa.append(np.mean(pa_error))

        # accel = self.compute_error_accel(np.array(gt_joints), np.array(pred_joints)).tolist()

        return abs_errors, errors, errors_pa, abs_pck, pck, accel


    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx14x3).
            joints_pred (Nx14x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        # (N-2)x14x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        return np.mean(normed[new_vis], axis=1)

    def LSPEvalHandler(self, premesh, gt_meshes, gt_joint, valids, is_joint=False):

        joints = np.matmul(self.J_regressor_LSP, premesh)

        gt_joint = np.matmul(self.J_regressor_LSP, gt_meshes)
        conf = None

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints * 1000
        abs_error, error, error_pa, abs_pck, pck, accel = self.compute_errors(gt_joint, joints, valids, confs=conf, format='lsp')

        return abs_error, error, error_pa, abs_pck, pck, accel

    def SMPLEvalHandler(self, premesh, gt_meshes, gt_joint, valids, is_joint=False):
        if is_joint:
            if premesh.shape[-1] == 3:
                joints = premesh
                conf = None
            elif premesh.shape[-1] == 4:
                joints = premesh[:,:,:3]
                conf = premesh[:,:,-1]
        else:
            joints = np.matmul(self.J_regressor_SMPL, premesh)
            conf = None

        gt_joint = np.matmul(self.J_regressor_SMPL, gt_meshes)

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck, accel = self.compute_errors(gt_joint, joints, valids, confs=conf, format='smpl')
        return abs_error, error, error_pa, abs_pck, pck, accel

    def PanopticEvalHandler(self, premesh, gt_joint, is_joint=False):
        joints = np.matmul(self.J_regressor_H36, premesh)
        conf = gt_joint[:,:,-1].copy()
        gt_joint = gt_joint[:,:,:3]
        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='h36m', confs=conf)
        return abs_error, error, error_pa, abs_pck, pck

    def MuPoTSEvalHandler(self, premesh, gt_joint, is_joint=False):
        h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        joints = np.matmul(self.J_regressor_H36, premesh)
        # gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        joints = joints[:,h36m_to_MPI]
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='mpi')

        if self.name == 'MPI3DHP':
            return abs_error, error, error_pa, abs_pck, pck
        else:
            return abs_error, error, error_pa, abs_pck, pck, joints

    def SMPLMeshEvalHandler(self, premeshes, gt_meshes, valid, is_joint=False):
        premeshes = premeshes * 1000
        gt_meshes = gt_meshes * 1000

        joints = np.matmul(self.J_regressor_LSP, premeshes)
        gt_joints = np.matmul(self.J_regressor_LSP, gt_meshes)

        vertex_errors = []

        premesh = self.align_mesh_by_pelvis_batch(premeshes, joints, format='lsp')
        gt_mesh = self.align_mesh_by_pelvis_batch(gt_meshes, gt_joints, format='lsp')

        vertex_errors = (np.mean(np.sqrt(np.sum((gt_mesh - premesh) ** 2, axis=-1)), axis=-1) * valid).tolist()

        return vertex_errors

    def evaluate(self, pred_meshes, gt_meshes, gt_joints, valids):
        abs_error, error, error_pa, abs_pck, pck, accel = self.eval_handler_mapper[self.name](pred_meshes, gt_meshes, gt_joints, valids)

        # calculate vertex error
        if gt_meshes.shape[1] < 6890:
            vertex_error = [None] * len(abs_error)
        else:
            # mesh in mm
            vertex_error = self.SMPLMeshEvalHandler(pred_meshes, gt_meshes, valids)

        return vertex_error, error, error_pa, abs_pck, pck, accel

    def eval_verts(self, pred_verts, gt_verts, valid=None):
        if valid is None:
            valid = np.ones((gt_verts.shape[0]), dtype=gt_verts.dtype)
        gt_joint = np.matmul(self.J_regressor_LSP, gt_verts)
        vertex_error, error, error_pa, abs_pck, pck, accel = self.evaluate(pred_verts, gt_verts, gt_joint, valid)
        return vertex_error, error, error_pa, accel

    def forward(self, pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, imgname, valids, is_seq):

        pred_poses = np.concatenate(pred_poses)
        pred_shapes = np.concatenate(pred_shapes)
        gt_poses = np.concatenate(gt_poses)
        gt_shapes = np.concatenate(gt_shapes)
        gt_joints = np.concatenate(gt_joints)
        valids = np.concatenate(valids)

        gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids = rearrange2seq(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids, imgname, is_seq)

        vertex_errors, errors, error_pas, abs_pcks, pcks, accels = [], [], [], [], [], []
        for gt_pose, gt_shape, pred_pose, pred_shape, gt_joint, valid in zip(gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids):
            trans = torch.zeros((gt_pose.shape[0], 3), dtype=torch.float32)
            if pred_shape.shape[1] == 6890: # For OOH
                pred_mesh = pred_shape
            else:
                pred_mesh, _ = self.smpl(torch.tensor(pred_shape, dtype=torch.float32), torch.tensor(pred_pose, dtype=torch.float32), trans)
                pred_mesh = pred_mesh.detach().cpu().numpy()
            if gt_pose.shape[1] > 1:
                gt_mesh, _ = self.smpl(torch.tensor(gt_shape, dtype=torch.float32), torch.tensor(gt_pose, dtype=torch.float32), trans)
                gt_mesh = gt_mesh.detach().cpu().numpy()
            else:
                gt_mesh = np.zeros((gt_pose.shape[0],1,3), dtype=np.float32)
            vertex_error, error, error_pa, abs_pck, pck, accel = self.evaluate(pred_mesh, gt_mesh, gt_joint, valid)
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            accels += accel

        if vertex_errors[0] is not None:
            vertex_error = np.mean(np.array(vertex_errors))
        else:
            vertex_error = -1
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        accel = np.mean(np.array(accels))

        return vertex_error, error, error_pa, abs_pck, pck, accel

    def calcu_loss(self, pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, imgname, valids, is_seq):

        pred_poses = np.concatenate(pred_poses)
        pred_shapes = np.concatenate(pred_shapes)
        gt_poses = np.concatenate(gt_poses)
        gt_shapes = np.concatenate(gt_shapes)
        gt_joints = np.concatenate(gt_joints)
        valids = np.concatenate(valids)

        gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids = rearrange2seq(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids, imgname, is_seq)

        vertex_errors, errors, error_pas, abs_pcks, pcks, accels = [], [], [], [], [], []
        for gt_pose, gt_shape, pred_pose, pred_shape, gt_joint, valid in zip(gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids):
            trans = torch.zeros((gt_pose.shape[0], 3), dtype=torch.float32)
            if pred_shape.shape[1] == 6890: # For OOH
                pred_mesh = pred_shape
            else:
                pred_mesh, _ = self.smpl(torch.tensor(pred_shape, dtype=torch.float32), torch.tensor(pred_pose, dtype=torch.float32), trans)
                pred_mesh = pred_mesh.detach().cpu().numpy()
            if gt_pose.shape[1] > 1:
                gt_mesh, _ = self.smpl(torch.tensor(gt_shape, dtype=torch.float32), torch.tensor(gt_pose, dtype=torch.float32), trans)
                gt_mesh = gt_mesh.detach().cpu().numpy()
            else:
                gt_mesh = np.zeros((gt_pose.shape[0],1,3), dtype=np.float32)
            vertex_error, error, error_pa, abs_pck, pck, accel = self.evaluate(pred_mesh, gt_mesh, gt_joint, valid)
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            accels += accel

        if vertex_errors[0] is not None:
            self.vertex_error += vertex_errors
        else:
            self.vertex_error = [-1]

        self.error += errors
        self.error_pa += error_pas
        self.abs_pck += abs_pcks
        self.pck += pcks
        self.accel += accels


    def calcu_interaction(self, gt_meshes, pred_meshes, inter_valid):
        valid = np.array(inter_valid).sum(axis=0)

        gt_joints_c1 = self.J_regressor_LSP @ gt_meshes[0].detach().cpu().numpy()
        gt_joints_c2 = self.J_regressor_LSP @ gt_meshes[1].detach().cpu().numpy()

        pred_joints_c1 = self.J_regressor_LSP @ pred_meshes[0].detach().cpu().numpy()
        pred_joints_c2 = self.J_regressor_LSP @ pred_meshes[1].detach().cpu().numpy()

        gt_joints_c1 = gt_joints_c1[valid == 2]
        gt_joints_c2 = gt_joints_c2[valid == 2]

        pred_joints_c1 = pred_joints_c1[valid == 2]
        pred_joints_c2 = pred_joints_c2[valid == 2]


        gt_joint_a, gt_joint_b = gt_joints_c1[:,:,None,:3], gt_joints_c2[:,None,:,:3]
        gt_interaction = np.linalg.norm(gt_joint_a - gt_joint_b, axis=-1)

        pred_joint_a, pred_joint_b = pred_joints_c1[:,:,None,:3], pred_joints_c2[:,None,:,:3]
        pred_interaction = np.linalg.norm(pred_joint_a - pred_joint_b, axis=-1)

        interaction = np.abs(gt_interaction - pred_interaction).mean(axis=(1,2))

        interaction = interaction * 1000.

        return interaction.tolist()

    def pair_by_L2_distance(self, alpha, gt_keps, src_mapper, gt_mapper, dim=17, gt_bbox=None):
        openpose_ant = []

        for j, gt_pose in enumerate(gt_keps):
            for i, pose in enumerate(alpha):
                diff = np.mean(np.linalg.norm(pose[src_mapper][:,:2] - gt_pose[gt_mapper][:,:2], axis=1) * gt_pose[gt_mapper][:,2])
                openpose_ant.append([i, j, diff, pose])

        iou = sorted(openpose_ant, key=lambda x:x[2])

        gt_ind = []
        pre_ind = []
        output = []
        # select paired data
        for item in iou:
            if (not item[1] in gt_ind) and (not item[0] in pre_ind):
                gt_ind.append(item[1])
                pre_ind.append(item[0])
                output.append([item[1], item[3]])

        if len(output) < 1:
            return None

        return gt_ind, pre_ind

    def mupots(self, pred_poses, pred_shapes, pred_trans, pred2ds, imgname, is_seq):
        pred_poses, pred_shapes, pred_trans, pred2ds, names = rearrange2frame(pred_poses, pred_shapes, pred_trans, pred2ds, imgname)
        import scipy.io as scio

        max_people = 3
        self.h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        self.halpe_to_MPI = [17,18,6,8,10,5,7,9,12,14,16,11,13,15,19,18,0]

        output_joints, output_joints2d = {}, {}
        for pred_pose, pred_shape, pred_tran, pred2d, name in zip(pred_poses, pred_shapes, pred_trans, pred2ds, names):
            pose = torch.from_numpy(pred_pose).float().to(torch.float32).reshape(-1, 72)
            shape = torch.from_numpy(pred_shape).float().to(torch.float32).reshape(-1, 10)
            trans = torch.from_numpy(pred_tran).float().to(torch.float32).reshape(-1, 3)
            verts, _ = self.smpl(shape, pose, trans)
            verts = verts.detach().cpu().numpy()

            joints = np.zeros((max_people, 17, 3))
            joints2d = np.zeros((max_people, 17, 3))

            for i, (vs, j2d) in enumerate(zip(verts, pred2d)):
                h36m_j3d = np.dot(self.J_regressor_H36, vs)
                joints[i] = h36m_j3d[self.h36m_to_MPI]
                joints2d[i] = j2d[self.halpe_to_MPI]

            name = name.split('/')[0]
            name = name.split('\\')[-1]

            if name in output_joints.keys():
                output_joints[name].append(joints)
            else:
                output_joints[name] = [joints]

            if name in output_joints2d.keys():
                output_joints2d[name].append(joints2d)
            else:
                output_joints2d[name] = [joints2d]


        for key in output_joints.keys():
            joints = np.array(output_joints[key]).reshape(-1, max_people, 17, 3)
            joints2d = np.array(output_joints2d[key]).reshape(-1, max_people, 17, 3)


            scio.savemat('output/%s.mat' %key, {'result': joints, 'result_2d': joints2d})

    def load_jta(self, imgname):
        import pickle
        img_name =  imgname[0].split('\\',6)[-1]
        data_folder = imgname[0].replace(img_name, '')
        dataset_annot = os.path.join(data_folder,'annot/test.pkl')

        with open(dataset_annot, 'rb') as f:
            params = pickle.load(f, encoding='iso-8859-1')
        
        joints_JTA, joints_2d_JTA, name_list = [], [], []
        for seq in params:
            if len(seq) < 1:
                continue
            for i, frame in enumerate(seq):
                frame_joints, frame_joints_2d = [], []
                name_list.append(os.path.join(data_folder, frame['img_path']))
                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue
                    frame_joints.append(np.array(frame[key]['JTA_joints_3d'], dtype=np.float32).reshape(-1,3))
                    frame_joints_2d.append(np.array(frame[key]['JTA_joints_2d'], dtype=np.float32).reshape(-1,3))
                joints_JTA.append(frame_joints)
                joints_2d_JTA.append(frame_joints_2d)
        
        return joints_JTA, joints_2d_JTA, name_list

    def jta(self, pred_poses, pred_shapes, pred_trans, gt_poses, gt_shapes, gt_joints, imgname, valids, is_seq):
        JTA_useful = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]
        Halpe2JTA_useful = [17, 18, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15]
        jta_idx = [i for i in range(14)]
        
        # metrics thresholds
        THS = [0.4, 0.8, 1.2]
        metrics_dict = {}
        for th in THS:
            for key in ['pr', 're', 'f1']:
                metrics_dict[f'{key}@{th}'] = []  
        
        # load jta annot
        jta_gt_joints, jta_gt_joints_2d, name_list = self.load_jta(imgname)

        pred_poses, pred_shapes, pred_trans, _, names = rearrange2frame(pred_poses, pred_shapes, pred_trans, pred_trans.copy(), imgname)
        
        for pred_pose, pred_shape, pred_tran, name in zip(pred_poses, pred_shapes, pred_trans, names):
            pose = torch.from_numpy(pred_pose).float().to(torch.float32).reshape(-1, 72)
            shape = torch.from_numpy(pred_shape).float().to(torch.float32).reshape(-1, 10)
            trans = torch.from_numpy(pred_tran).float().to(torch.float32).reshape(-1, 3)
            verts, _ = self.smpl(shape, pose, trans)
            verts = verts.detach().cpu().numpy()
            
            joints = np.matmul(self.J_regressor_Halpe, verts)
            pred_joints = joints[:, Halpe2JTA_useful]
            pred_joints = np.insert(pred_joints, 0, values=jta_idx, axis=2)
            pred_joints = np.concatenate(pred_joints, axis=0)

            # get gt data according to image name
            frame_id = name_list.index(name)
            jta_gt_joint = np.array(jta_gt_joints[frame_id])
            jta_gt_joint_2d = np.array(jta_gt_joints_2d[frame_id])

            gt_joints_use = jta_gt_joint[:, JTA_useful]
            gt_joints_use = np.insert(gt_joints_use, 0, values=jta_idx, axis=2)
            gt_joints_use = np.concatenate(gt_joints_use, axis=0)
            
            gt_2d = jta_gt_joint_2d[:, JTA_useful]
            gt_2d = np.concatenate(gt_2d, axis=0)
            idx = np.where((gt_2d[:,0] < 0) | (gt_2d[:,0] > 1920 | (gt_2d[:,1] < 0) | (gt_2d[:,1] > 1080)))
            gt_joints_use = np.delete(gt_joints_use, idx, axis=0)

            
            for th in THS:
                __m = joint_det_metrics(points_pred=pred_joints, points_true=gt_joints_use, th=th)
                for key in ['pr', 're', 'f1']:
                    metrics_dict[f'{key}@{th}'].append(__m[key])
        
        for th in THS:
            for key in ['pr', 're', 'f1']:
                metrics_dict[f'{key}@{th}'] = np.mean(metrics_dict[f"{key}@{th}"]) * 100
        
        vertex_error, error, error_pa, abs_pck, pck, accel = self.forward(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, imgname, valids, is_seq)    
        
        return metrics_dict, vertex_error, error, error_pa, abs_pck, pck, accel

