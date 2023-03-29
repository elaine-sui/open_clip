"""
Majority of code from https://github.com/microsoft/VSE_Gradient/blob/main/Gradient/gradient.py
"""

import torch
import torch.nn.functional as F

def weight_triplet_nca(Pos, Neg, tau=10, *args):
    PN = Pos-Neg # similarity difference
    return (1/(1+torch.exp(tau * PN))).unsqueeze(1)

def weight_pair_con(Pos, Neg, param_dict):
    wp = torch.ones_like(Pos)
    wn = torch.ones_like(Neg)
    return wp.unsqueeze(1), wn.unsqueeze(1)

# gradient direction
def grad_dir_cos(fvec_img, fvec_txt, Neg_img_idx, Neg_txt_idx, wp_it, wn_it, wp_ti, wn_ti, wt_it, wt_ti):
    G_img = wt_it * (wn_it*fvec_txt[Neg_txt_idx,:] - wp_it*fvec_txt ) - wt_ti * ( wp_ti*fvec_txt ) # G_anc_img = txt + txt feature = txt feautre
    G_txt = wt_ti * (wn_ti*fvec_img[Neg_img_idx,:] - wp_ti*fvec_img ) - wt_it * ( wp_it*fvec_img ) # G_anc_txt = img + img feature = img feautre
    
    N = fvec_img.size(0)
    
    for i in range(N):           
        index_neg = Neg_img_idx[i]
        G_img[index_neg,:] += wt_ti[i]*wn_ti[i]*fvec_txt[i,:]
        
    for i in range(N):           
        index_neg = Neg_txt_idx[i]
        G_txt[index_neg,:] += wt_it[i]*wn_it[i]*fvec_img[i,:]

    return G_img, G_txt

def weight_triplet(mode):
    if mode=='nca':
        return weight_triplet_nca

def weight_pair(mode):
    if mode=='con':
        return weight_pair_con

def grad_dir(mode):
    # gradient distance
    if mode=='cos':
        return grad_dir_cos

def gap_dir(fvec_img, fvec_txt):
    gap = F.normalize((F.normalize(fvec_img) - F.normalize(fvec_txt)))

    return gap, -gap

