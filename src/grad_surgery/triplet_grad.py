"""
Majority of code from https://github.com/microsoft/VSE_Gradient/blob/main/Gradient/gradient.py
"""

import torch.nn as nn

from .utils import weight_triplet, weight_pair, grad_dir
from .triplet_grad_fn import TripletGradFun

class TripletGrad(nn.Module):
        
    def __init__(self, gmode=('nca', 'con', 'cos'), tau=1, ms_a=2, ms_b=10, ms_l=0.5, grad_t=1., gap_t=0.1):
        super(TripletGrad, self).__init__()
        # gradient mode
        mode_tp, mode_pr, mode_gd = gmode

        # rs: relative similarity
        fun_rs = None
        
        self.fun_dict = {'wt': weight_triplet(mode_tp),
                         'wp': weight_pair(mode_pr),
                         'rs': fun_rs,
                         'gd': grad_dir(mode_gd),
                         }
        
        self.param_dict = {'tau': tau,
                           'ms_a': ms_a,
                           'ms_b': ms_b,
                           'ms_l': ms_l,
                           'grad_t': grad_t,
                           'gap_t': gap_t
                           }

        self.gradfun = TripletGradFun()

    def forward(self, input_img, input_txt):
        return self.gradfun.apply(input_img, input_txt, self.fun_dict, self.param_dict)