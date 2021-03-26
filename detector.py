# -*- coding: utf-8 -*-
import cv2
import time
import torch
import numpy as np
from utils import get_affine_transform,_transpose_and_gather_feat, _nms, _topk, _topk_channel, multi_pose_post_process
from show_all_imgs import add_coco_hp

num_classes = 1
vis_thresh = 0.3

# mean & std value of per pictrue
mean = [0.408, 0.447, 0.470]
std = [0.289, 0.274, 0.278]

def pre_process(image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)

    inp_height, inp_width = 416, 416
    c = np.array([new_width / 2., new_height / 2.])
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    images = inp_image
    meta = {'c': c,'s': s, 'out_height': inp_height //4, 'out_width': inp_width // 4}
    return images, meta

#==================================================================================

def multi_pose_decode(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    #>>>:param heat: keypoint heatmap 定位目标中心点的heatmap : (1, 1, 128, 128)
    #    :param wh: object size 确定矩形宽高  : (1, 2, 128, 128)
    #>>>:param kps: joint locations 相对于目标中心的各关键点偏移 : (1, 34, 128, 128)
    #    :param reg: local offset 包围框的偏移补偿  : (1, 2, 128, 128)
    #>>>:param hm_hp: joint heatmap 一般的关键点估计heatmap : (1, 17, 128, 128)
    #    :param hp_offset: joint offset 关键点估计的偏移 : (1, 2, 128, 128)
    #    :param K: top-K :100

	# batch:1  cat:1(类别数)  height:128  width:128
    _, cat, height, width = heat.size()
    
    heat = heat.reshape([-1, height, width])
    wh = wh.reshape([-1, height, width])
    kps = kps.reshape([-1, height, width])
    reg = reg.reshape([-1, height, width])
    hm_hp = hm_hp.reshape([-1, height, width])
    hp_offset = hp_offset.reshape([-1, height, width])

    num_joints = 17 
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    
    kps = _transpose_and_gather_feat(kps, inds)     
    kps[..., ::2] += xs.view(  K, 1).expand(  K, num_joints)
    kps[..., 1::2] += ys.view(  K, 1).expand(  K, num_joints)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        xs = xs.view(  K, 1) + reg[ :, 0:1]        
        ys = ys.view(  K, 1) + reg[ :, 1:2]        
    else:
        xs = xs.view(  K, 1) + 0.5
        ys = ys.view(  K, 1) + 0.5         
    
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(  K, 2)

    scores = scores.view(  K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=1)

    if hm_hp is not None:
        hm_hp_nms_time = time.time()        
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view( K, num_joints, 2).permute( 1, 0, 2).contiguous() # J x K x 2
        reg_kps = kps.unsqueeze(2).expand(  num_joints, K, K, 2)
        
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # J x K        
        
        hm_inds = hm_inds.reshape([17*K])
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset, hm_inds)
            hp_offset = hp_offset.view(  num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[ :, :, 0]          
            hm_ys = hm_ys + hp_offset[ :, :, 1]
            
        else:
            hm_xs = hm_xs + 0.4
            hm_ys = hm_ys + 0.4
        
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(1).expand(  num_joints, K, K, 2)
        
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=3) ** 0.5)  # [17, k, k]        
        min_dist, min_ind = dist.min(dim=2) # J x K  [17,8]        
        hm_score = hm_score.gather(1, min_ind).unsqueeze(-1) # J x K x 1  [17, 8, 1]
        
        min_dist = min_dist.unsqueeze(-1)  # [17, 8, 1]        
        min_ind = min_ind.view(  num_joints, K, 1, 1).expand(  num_joints, K, 1, 2)  # [17, 8, 1, 2]
        
        hm_kps = hm_kps.gather(2, min_ind)  #  [17, 8, 1, 2]        
        hm_kps = hm_kps.view(  num_joints, K, 2)  #  [17, 8, 1, 2]
        
        l = bboxes[ :, 0].view(  1, K, 1).expand(  num_joints, K, 1)
        t = bboxes[ :, 1].view(  1, K, 1).expand(  num_joints, K, 1)
        r = bboxes[ :, 2].view(  1, K, 1).expand(  num_joints, K, 1)
        b = bboxes[ :, 3].view(  1, K, 1).expand(  num_joints, K, 1)
        
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        
        mask = (mask > 0).float().expand(  num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute( 1,0, 2).contiguous().view(  K, num_joints * 2)
    # detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    detections = torch.cat([bboxes, scores, kps, torch.transpose(hm_score.squeeze(dim=2), 0, 1)], dim=1)

    return detections.numpy()
    
#===============================================================================
# post process of decode output
def post_process(dets, meta, scale=1):
    dets = multi_pose_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])
    dets = np.array(dets, dtype=np.float32).reshape(-1, 56) # 0:3,x1,y1;x2,y2 4:中心点置信度 5:39,keypoint 40:56, keypoint 置信度
    return dets

# ===============================================================================
# show results
def show_results(image, results):
    for b_id, detection in enumerate(results):        
        bbox = detection[:4]
        bbox_prob = detection[4]
        keypoints = detection[5:39]
        keypoints_prob = detection[39:]
        if bbox_prob > vis_thresh:           
            add_coco_hp(image, keypoints, keypoints_prob)
            
            
            
