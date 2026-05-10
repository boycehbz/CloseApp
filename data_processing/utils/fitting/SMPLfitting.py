'''
 @FileName    : SMPLfitting.py
 @EditTime    : 2022-03-15 21:48:44
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from utils.fitting.utils import init, init_guess
from utils.fitting.non_linear_solver import non_linear_solver

class SMPLfitting():
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        self.setting = init()
    
    def __call__(self, mesh):
        init_guess(self.setting, mesh)
        params, mesh = non_linear_solver(self.setting, mesh)

        return params['pose'], params['betas'], params['transl']