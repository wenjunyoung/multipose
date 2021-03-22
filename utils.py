# -*- coding: utf-8 -*-  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import torch
import time
#==================================================================
#  pre-process image
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

#============================================================
#   decode
def _gather_feat(feat, ind, mask=None):
    
    dim  = feat.size(1)
    ind  = ind.unsqueeze(1).expand(ind.size(0), dim)
    feat = feat.gather(0, ind)
    
    if mask is not None:
        
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute( 1, 2, 0).contiguous()
    feat = feat.view( -1, feat.size(2))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    hmax = torch.nn.functional.max_pool2d( heat, (3, 3), stride=1, padding=1)
    keep = (hmax == heat).float()
    return heat * keep

def _topk_channel(scores, K=40):
	# cat:17  height:128  width:128    
    cat, height, width = scores.size()
    # top_scores: [17, 100]
    # topk_inds: [17, 100]
    topk_scores, topk_inds = torch.topk(scores.view( cat, -1), K)  # ([1, 17, 16384])
    topk_inds = topk_inds % (height * width)

    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
	# batch:1  cat:1(类别数)  height:128  width:128      
    cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(cat, -1), K) 
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(-1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view( -1, 1), topk_ind).view(K)
    topk_ys = _gather_feat(topk_ys.view( -1, 1), topk_ind).view(K)
    topk_xs = _gather_feat(topk_xs.view( -1, 1), topk_ind).view(K)   
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

# ====================================================================================
# post process
def multi_pose_post_process(dets, c, s, h, w):
    # dets: k x 56
    # return list of 39 in image coord

    bbox = transform_preds(dets[ :, :4].reshape(-1, 2), c[0], s[0], (w, h))
    pts = transform_preds(dets[:, 5:39].reshape(-1, 2), c[0], s[0], (w, h))
    top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[:, 4:5], 
            pts.reshape(-1, 34), dets[:, 39:56]], axis=1).astype(np.float32).tolist()

    return top_preds
