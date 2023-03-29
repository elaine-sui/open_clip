"""
Majority of code from https://github.com/microsoft/VSE_Gradient/blob/main/Gradient/gradient.py
"""

import torch
from utils import gap_dir

class TripletGradFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, fvec_img, fvec_txt, fun_dict, param_dict):
 
        ############################################
        # preprocessing
        N = fvec_img.size(0)
        
        # Pos value
        Pos = torch.mul(fvec_img, fvec_txt).sum(1)

        # Similarity Matrix
        scores = torch.mm(fvec_img, fvec_txt.t()).fill_diagonal_(-1)

        # Neg value
        Neg_txt, Neg_txt_idx = scores.max(1)
        Neg_img, Neg_img_idx = scores.max(0)

        ############################################
        # triplet weight for image
        wt_it = fun_dict['wt'](Pos, Neg_txt, param_dict['tau'])
        wt_ti = fun_dict['wt'](Pos, Neg_img, param_dict['tau'])

        # pair weight for image
        wp_it, wn_it = fun_dict['wp'](Pos, Neg_txt, param_dict)
        wp_ti, wn_ti = fun_dict['wp'](Pos, Neg_img, param_dict)
        
        # gradient direction->txt, img, f_anc_txt, f_anc_img
        G_img, G_txt = fun_dict['gd'](fvec_img, fvec_txt, Neg_img_idx, Neg_txt_idx, wp_it, wn_it, wp_ti, wn_ti, wt_it, wt_ti)
        
        gap_img, gap_txt = gap_dir(fvec_img, fvec_txt)
        G_img += param_dict['gap_t'] * gap_img
        G_txt += param_dict['gap_t'] * gap_txt

        ctx.save_for_backward(G_img, G_txt)
        
        ############################################
        # origin triplets
        T_it = (Neg_txt - Pos).sum()
        T_ti = (Neg_img - Pos).sum()
        
        # loss
        loss = T_it+T_ti

        loss_log = {"loss_it": T_it, 
                    "loss_ti": T_ti,
                    "positive": Pos, 
                    "negative_txt": Neg_txt,  
                    "negative_img": Neg_img, 
                    }
            
        return loss, loss_log
    
    @staticmethod
    def backward(ctx, grad_output, no_use):
        
        G_img, G_txt = ctx.saved_tensors
        return G_img, G_txt, None, None