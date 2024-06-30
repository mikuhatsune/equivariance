# from depth_metrics import evaluate_depth_metrics, accum_depth_metrics, average_depth_metrics, DepthMetrics
 ## https://github.com/SysCV/P3Depth/blob/main/src/metrics.py
import dataclasses
import math
from typing import Tuple, List
import torch
import numpy as np
from torch import Tensor

@dataclasses.dataclass
class DepthMetrics(object):
    silog: float = 0.
    rmse: float = 0.
    rmse_log: float = 0.
    sq_rel: float = 0.
    abs_rel: float = 0.
    lg10: float = 0.
    delta1: float = 0.
    delta2: float = 0.
    delta3: float = 0.
    n: int = 0

def accum_depth_metrics(accum, new):
    if accum is None:
        return new
    n = new.n
    accum.silog += new.silog * n
    accum.rmse += new.rmse * n
    accum.rmse_log += new.rmse_log * n
    accum.sq_rel += new.sq_rel * n
    accum.abs_rel += new.abs_rel * n
    accum.lg10 += new.lg10 * n
    accum.delta1 += new.delta1 * n
    accum.delta2 += new.delta2 * n
    accum.delta3 += new.delta3 * n
    accum.n += n
    return accum

def average_depth_metrics(accum):
    n = accum.n
    return DepthMetrics(
        silog=accum.silog / n,
        rmse=accum.rmse / n,
        rmse_log=accum.rmse_log / n,
        sq_rel=accum.sq_rel / n,
        abs_rel=accum.abs_rel / n,
        lg10=accum.lg10 / n,
        delta1=accum.delta1 / n,
        delta2=accum.delta2 / n,
        delta3=accum.delta3 / n,
    )


def compute_errors(pred_disp, gt, min_depth_eval, max_depth_eval):
    # pred = recover_metric_depth(pred, gt)
    x = pred_disp; y = 1. / gt
    xm = x.mean(); x0 = x - xm
    ym = y.mean(); y0 = y - ym
    beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
    pred = x0 * beta + ym
    pred = (1 / pred).clip(min_depth_eval, max_depth_eval)

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    if np.isnan(rmse_log):
        print(gt, pred, gt.shape)

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


# def recover_metric_depth(pred, gt):
#     if type(pred).__module__ == torch.__name__:
#         pred = pred.cpu().numpy()
#     if type(gt).__module__ == torch.__name__:
#         gt = gt.cpu().numpy()

#     # print(gt,pred)
#     # gt_mean = np.mean(gt)
#     # pred_mean = np.mean(pred)
#     # pred_metric = pred * (gt_mean / pred_mean)
#     # return pred_metric
    
#     # gt_mean = gt.mean(axis=[1,2])
#     # pred_mean = pred.mean(axis=[1,2])
#     x = pred; y = gt
#     # print(x.shape)
#     xm = x.mean(); x0 = x - xm
#     ym = y.mean(); y0 = y - ym
#     beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
#     p = x0 * beta + ym
#     return p


def evaluate_depth_metrics(pred_disp, gt_depth, dataset_type, max_depth=10, batch=True) -> DepthMetrics:
    pred_disp = pred_disp.cpu().numpy()
    gt_depth = gt_depth.cpu().numpy()

    min_depth_eval = 1e-3
    max_depth_eval = max_depth
    
    # pred_disp = pred_disp.clip(1/max_depth_eval, 1/min_depth_eval)
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    # ## Eigen eval
    # if "KITTI" in dataset_type:
    #     _, _, gt_height, gt_width = _target.shape
    #     eval_mask = np.zeros(valid_mask.shape)
    #     eval_mask[ :, :,int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
    #     # eval_mask[ :, :, int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1 # GARG CROP
    #     valid_mask = np.logical_and(valid_mask, eval_mask)

    n = len(gt_depth)
    if batch:
        silog = log10 = abs_rel = sq_rel = rmse = rmse_log = d1 = d2 = d3 = 0.
        for i in range(n):
            _silog, _log10, _abs_rel, _sq_rel, _rmse, _rmse_log, _d1, _d2, _d3=compute_errors(
                pred_disp[i][valid_mask[i]], gt_depth[i][valid_mask[i]],
                min_depth_eval, max_depth_eval)
            silog += _silog; log10 += _log10; abs_rel += _abs_rel
            sq_rel += _sq_rel; rmse += _rmse; rmse_log += _rmse_log
            d1 += _d1; d2 += _d2; d3 += _d3
        silog /= n; log10 /= n; abs_rel /= n
        sq_rel /= n; rmse /= n; rmse_log /= n
        d1 /= n; d2 /= n; d3 /= n
    else:
         silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3=compute_errors(
             pred_disp[valid_mask], gt_depth[valid_mask],
             min_depth_eval, max_depth_eval)

    metrics = DepthMetrics(
        silog=float(silog),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        sq_rel=float(sq_rel),
        abs_rel=float(abs_rel),
        lg10=float(log10),
        delta1=float(d1),
        delta2=float(d2),
        delta3=float(d3),
        n=n,
    )

    return metrics