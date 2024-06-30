from scipy.io import loadmat
import h5py
import numpy as np
import torch

class NyuDepthV2(torch.utils.data.Dataset):
#     # download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
#     NYU_DATA_PATH = "nyu_depth_v2_labeled.mat"

#     # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
#     NYU_SPLIT_PATH = "splits.mat"

#     # download from https://drive.google.com/file/d/1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC
#     MODEL_PATH = "model.pt"

    def __init__(self,
                 datapath='/u/yuanyiz2/work/nyu_data/nyu_depth_v2_labeled.mat',
                 splitpath='/u/yuanyiz2/work/nyu_data/splits.mat',
                 split="test", transform=None):

        self.__image_list = []
        self.__depth_list = []

        self.__transform = transform

        mat = loadmat(splitpath)

        if split == "train":
            indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
        elif split == "test":
            indices = [ind[0] - 1 for ind in mat["testNdxs"]]
        else:
            raise ValueError("Split {} not found.".format(split))

        with h5py.File(datapath, "r") as f:
            for ind in indices:
                self.__image_list.append(np.swapaxes(f["images"][ind], 0, 2))
                self.__depth_list.append(np.swapaxes(f["rawDepths"][ind], 0, 1))

        self.__length = len(self.__image_list)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = self.__image_list[index]
        image = image / 255

        # depth
        depth = self.__depth_list[index]

        # mask; cf. project_depth_map.m in toolbox_nyu_depth_v2 (max depth = 10.0)
        mask = (depth > 0) & (depth < 10)

        # sample
        sample = {}
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = mask

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample

midas_root = '../MiDaS/'
import sys
sys.path.insert(0, midas_root)

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
# from midas.transforms import Resize, NormalizeImage, PrepareForNet

dev = 0


import sys
import os
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms, utils, models
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from matplotlib import patches

import argparse

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--dir', type=str, default='logs')
parser.add_argument('--tag', type=str,
                    # default='midas_nyu_finetune_unsup',
                    # default='debug',
                    default='dpt_nyu_finetune2',
                   )
parser.add_argument('--task', type=str,
                    default='depth'
                    # default='depth_zbuffer'
                    # default='edge_texture'
                   )
# parser.add_argument('--split', type=str, default='fast')
parser.add_argument('--num_bldg', type=int,
                    # default=3,
                    default=100,
                   )
# parser.add_argument('--percent', type=float, default=33)
parser.add_argument('--label', type=int, default=None)
parser.add_argument('--unlabel', type=int, default=0)
parser.add_argument('--unl_batch', type=int, default=1)
parser.add_argument('--val', type=str,
                    default='tiny',
                    # default='c'
                   )
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--k', type=int,
                    default=3
                    # default=1
                   )
parser.add_argument('--pos_encode', type=int, default=1)
# parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--box_type', type=str,
                    default='s0.4-1,r,op0.2'
                    # default='s0.6-1.1,r,op0.1'
                    # default='s1-1,op0'
                   )
# parser.add_argument('--var_reduct', type=int, default=0)
parser.add_argument('--sup_coef', type=float,
                    default=1
                    # default=0
                   )
parser.add_argument('--a_coef', type=float, default=0)
parser.add_argument('--equi_coef', type=float, default=0.0001)
parser.add_argument('--unl_coef', type=float, default=0.0001)
parser.add_argument('--equi_q', type=int, default=0,
                    help='imp samp q: 0-no, 1-learn q IS dist that favors hard crops')
parser.add_argument('--q_lr', type=float, default=0.0)
parser.add_argument('--load_q', type=str, default=None)
# parser.add_argument('--equi_sg', type=int, default=0, help='stop_grad on avg target')
parser.add_argument('--equi_loss_mode', type=str, default='forward',
                    help='loss form: forward |f*t - t*fa|, backward |fa - t-1*f*t|')
# parser.add_argument('--equi_layers', type=str,
#                     default='last_conv2',
# #                     default='last_conv1',
#                     help='last_conv2,last_conv1,up_blocks')
parser.add_argument('--net', type=str,
                    # default='midas_v21',
                    default='dpt_large')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dev', type=str, default='0')
parser.add_argument('--val0', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--init', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--train_fp16', type=int, default=1)
parser.add_argument('--val_fp16', type=int, default=0)
parser.add_argument('--sep_batch', type=int, default=0)
parser.add_argument('--test_only', type=int, default=0)
parser.add_argument('--color_aug', type=int,
                    # default=0
                    default=1
                   )
parser.add_argument('--color_strength', type=float, default=0.4)
parser.add_argument('--hue_ratio', type=float, default=0.25)
parser.add_argument('--rgb_norm', type=int, default=1)
parser.add_argument('--val_color_aug', type=int, default=0)
parser.add_argument('--predictor', type=int,
                   default=0,
                   # default=1
                   )
parser.add_argument('--tb', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)
args = parser.parse_args([])


# mean_=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1); std_=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
# mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)  # cifar10
# mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)  # imagenet

@torch.no_grad()
def rescale(x):
    a, b = x.min(), x.max()
    return (x-a)/(b-a)
@torch.no_grad()
def pt2np(x, denorm=False):
    x = x.cpu()
    if denorm:
        x = x*std_+mean_
    if x.ndim >= 3:
        x = x.permute(1,2,0)
    return x.numpy()

task = args.task
if task == 'edge_texture':
    has_mask = False; out_dim = 1
elif task == 'keypoints3d':
    has_mask = False; out_dim = 1
elif task == 'segment_semantic':
    has_mask = True; out_dim = 17
elif task == 'normal':
    has_mask = True; out_dim = 3
elif task == 'depth_zbuffer' or task == 'depth':
    has_mask = True; out_dim = 1
else:
    raise ValueError(f'unsupported task: {task}')
batch_size=8  #32
num_workers=10


# if args.tag.endswith('.pt'):
#     args.tag = os.path.dirname(args.tag)
savedir = f'{args.dir}/{task}_{args.tag}/'
os.makedirs(savedir, exist_ok=True)
file = open(savedir+f'_log.txt', 'a')
def printf(*args, fileonly=False, **kwargs):
    if not fileonly:
        print(*args, **kwargs)
    print(*args, **kwargs, file=file)
    file.flush()


default_models = {
    "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
}
model_path = midas_root + default_models[args.net]

if args.net == 'midas_v21':
    # model = MidasNet(model_path, non_negative=True)
    mean_=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1); std_=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

elif args.net == 'dpt_large':
    # model = DPTDepthModel(
    #     path=model_path,
    #     backbone="vitl16_384",
    #     non_negative=True,
    # )
    mean_=torch.tensor([0.5, 0.5, 0.5]).view(3,1,1); std_=torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)


class WeightedBoxSampler:
    def __init__(self, scale=(0.08,1.0), ratio=(0.75,4./3), out_portion=None,
                 bins_scale=3, bins_ratio=3, bins_xy=3):
        self.scale = scale
        self.log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        self.out_portion = out_portion

        self.bins_scale = bins_scale
        self.bins_ratio = bins_ratio
        self.bins_xy = bins_xy
        # self.q = torch.ones((bins_scale, bins_ratio, bins_xy, bins_xy))
        if args.load_q:
            self.q = torch.from_numpy(np.load(args.load_q))
            printf(f'load_q from: {args.load_q}')
        else:
            self.q = torch.ones(bins_scale*bins_ratio*bins_xy*bins_xy)

        self.scales = np.linspace(self.scale[0], self.scale[1], bins_scale+1)
        self.log_ratios = np.linspace(self.log_ratio[0], self.log_ratio[1], bins_ratio+1)

    def __call__(self, height, width, imp=True):
        q = self.q
        for trial in range(10):
            if imp:
                iq = torch.multinomial(q, 1).item()
            else:
                iq = torch.randint(len(q), ()).item()
            ia = iq
            ix = ia % self.bins_xy; ia //= self.bins_xy
            iy = ia % self.bins_xy; ia //= self.bins_xy
            ir = ia % self.bins_ratio; ia //= self.bins_ratio
            ia = ia % self.bins_ratio

            scale0, scale1 = self.scales[ia], self.scales[ia+1]
            lr0, lr1 = self.log_ratios[ir], self.log_ratios[ir+1]

            a = height*width * torch.empty(1).uniform_(scale0, scale1).item()
            r = torch.empty(1).uniform_(lr0, lr1).exp_().item() * width/height
            h = int(round(np.sqrt(a / r)))  # rand_round
            w = int(round(np.sqrt(a * r)))

            o = self.out_portion  # 0 <= out_portion <= 1 of the crop length
            if isinstance(o, float):
                # make the center of boxes roughly uniform
                i = int(round(-h*o + (iy + torch.rand(()).item()) * (height - (1-2*o)*h) / self.bins_xy))
                j = int(round(-w*o + (ix + torch.rand(()).item()) * (width - (1-2*o)*w) / self.bins_xy))
            elif o == 'pad':
                # close to the pytorch version, prefer boxes within boundary, but allow padding 0
                i = iy if h <= height else iy-self.bins_xy
                j = ix if w <= width else ix-self.bins_xy
                i = int(round( (i + torch.rand(()).item()) * abs(height - h) / self.bins_xy ))
                j = int(round( (j + torch.rand(()).item()) * abs(width - w) / self.bins_xy ))
            else:
                # the pytorch version, guarantee boxes with boundary, reject outside boxes (no padding)
                if h > height or w > width:
                    # if trial > 5: print(trial, a, r, h, w)
                    if trial == 10-1:
                        i, j, h, w = 0, 0, height, width
                    else:
                        continue
                else:
                    i = int(round( (iy + torch.rand(()).item()) * abs(height - h) / self.bins_xy ))
                    j = int(round( (ix + torch.rand(()).item()) * abs(width - w) / self.bins_xy ))
            break

        return i, j, h, w, iq, q[iq]

    def __repr__(self):
        return (f"{self.__class__.__name__}"
            "(scale={scale}, log_ratio={log_ratio}, out_portion={out_portion}, "
            "bins_scale={bins_scale}, bins_ratio={bins_ratio}, bins_xy={bins_xy}").format(**self.__dict__)


class DictAug:
    # size: (h, w)
    def __init__(self, size=(288,384), k=3, pos_encode=0, box_type='s0.4-1,r,op0.2'):
        self.size = size

        box_type = box_type.split(',')
        scale = (1., 1.)
        ratio = (1., 1.)
        for b in box_type:
            if b[0] == 's':
                scale = box_type[0][1:].split('-')
                scale = (float(scale[0]), float(scale[1]))
            elif b == 'r':
                ratio = (3./4, 4./3)
            elif b == 'torch':
                out_portion = None
            elif b.startswith('op'):
                out_portion = float(b[2:])
            else:
                raise ValueError(f'unsupported box_type: {b}')

        self.sampler = WeightedBoxSampler(scale=scale, ratio=ratio, out_portion=out_portion)
        self.k = k
        self.pos_encode = pos_encode
        h, w = size
        if self.pos_encode:
            self.pos = torch.empty((2, h, w))
            x = torch.linspace(0, 1, w)
            y = torch.linspace(0, 1, h)
            self.pos[0] = x.view(1,-1)
            self.pos[1] = y.view(-1,1)
            self.pos = self.pos.to(dev)
        self.default_mask = torch.full((1,1,h,w), 2.0, device=dev)

        if args.color_aug:
            strength = args.color_strength
            self.color_aug = transforms.ColorJitter(
                brightness=strength,
                contrast=strength,
                saturation=strength,
                hue=args.hue_ratio*strength)
            printf(f'use color aug: [{strength},{strength},{strength},{args.hue_ratio*strength}]')
        if args.rgb_norm:
            self.norm = transforms.Normalize(mean=mean_.view(-1), std=std_.view(-1))
            printf(f'use rgb norm: {self.norm}')
        else:
            self.norm = lambda x: x

    def __call__(self, d, imp=True, color_aug=args.color_aug):
        # (B,3,H,W), (B,?,H,W), (B,1,H,W)
        task = 'depth'
        rgbs = d['image'].to(dev, non_blocking=True)
        targets = d[task].to(dev, non_blocking=True)
        # rgb = rgb.view(-1, *rgb.shape[2:])
        n = len(rgbs)
        if 'mask' in d:
            masks = d['mask'].to(dev, non_blocking=True) + 1
        else:
            masks = self.default_mask.expand(n,-1,-1,-1)

        assert isinstance(rgbs, torch.Tensor)
        h0, w0 = rgbs.shape[-2:]  # PIL image.size is (width, height), pytorch.shape is (height, width)

        # randomly make k crops for each image
        list_boxes = []; qs = torch.empty((n,self.k,2), dtype=torch.float32, device=dev)
        for b in range(n):
            boxes = [(0, 0, self.size[1], self.size[0])]  # size=(h,w) -> (w,h)
            qs[b,0,0], qs[b,0,1] = -1, 1.
            for _ in range(1,self.k):
                i, j, h, w, qs[b,_,0], qs[b,_,1] = self.sampler(*self.size, imp=imp)
                box = j, i, j+w, i+h  # roi_align takes x0,y0,x1,y1 format, but interpolate takes y,x,h,w
                boxes.append(box)
            list_boxes.append(torch.tensor(boxes, dtype=torch.float32, device=dev))

        if not color_aug and args.rgb_norm:
            rgbs = self.norm(rgbs)
        stack = [rgbs]
        if self.pos_encode:
            stack += [self.pos[None].expand(n,-1,-1,-1)]
        stack += [targets]
        if masks is not None:
            stack += [masks.float()]
        stack = torch.cat(stack, dim=1)

        crops = torchvision.ops.roi_align(stack, list_boxes, self.size, aligned=True)
        if color_aug:
            for _ in  range(len(crops)):
                if color_aug == 2 and _ % self.k == 0:
                    if args.rgb_norm:
                        crops[_,:3] = self.norm(crops[_,:3])
                else:
                    crops[_,:3] = self.norm(self.color_aug(crops[_,:3]))

        _ = 3 + (len(self.pos)*self.pos_encode if self.pos_encode else 0)
        r = {'rgb': crops[:, :_], task: crops[:, _:_+targets.shape[1]], 'box': list_boxes, 'q': qs,
            # 'orig_depth': targets, 'orig_mask': masks
            }
        if masks is not None:
            r['mask'] = crops[:, -1:] - 1
        return r

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, k={self.k}, pos={self.pos_encode}, sampler={self.sampler})"


def cos_window(size, h):
    t = torch.linspace(0,torch.pi/2,h)
    s = torch.sin(t)
    win = torch.ones(size)
    # sides
    win[:h,:] = s[:,None]
    win[-h:,:] = s[:,None].flip(0)
    win[:,:h] = s[None,:]
    win[:,-h:] = s[None,:].flip(1)
    # corners
    t = torch.pi/2 - torch.sqrt(t[:,None]**2 + t[None,:]**2)
    s = torch.sin(t.clip_(min=0))
    win[:h,:h] = s.flip(0,1)
    win[:h,-h:] = s.flip(0)
    win[-h:,:h] = s.flip(1)
    win[-h:,-h:] = s
    return win

# win = cos_window((32,32), 5)
win = cos_window((128,128), 20).to(dev)
# plt.imshow(win); plt.show()


def average(crops, boxes, target_size, window=None, skip_win_for_0=True,
            q=None, ret_resized=False):
    boxes = boxes.int().cpu().numpy()  # will this be faster than gpu?

    n = len(crops)
    dev = crops.device
    a_img = torch.zeros((crops.shape[1],) + target_size, device=dev)
    a_w = torch.zeros((1,) + target_size, device=dev)
    target_h, target_w = target_size

    if window is not None:
        window = window.view(1,1,*window.shape[-2:])

    res = []
    for i, crop, xyxy in zip(range(n), crops, boxes):
        x,y,x1,y1 = xyxy
        w, h = x1-x, y1-y
        # resized = TF.resize(crop, [h,w], TF.InterpolationMode.BILINEAR, antialias=antialias)
        # resized = F.interpolate(crop[None], (h,w), mode='nearest')[0]
        # resized = F.interpolate(crop[None], (h,w), mode='nearest-exact')[0]
        # resized = F.interpolate(crop[None], (h,w), mode='area')[0]
        # resized = F.interpolate(crop[None], (h,w), mode='bilinear')[0]
        resized = F.interpolate(crop[None], (h,w), mode='bilinear', align_corners=True)[0]
        if x < 0 or y < 0 or x1 > target_w or y1 > target_h:
            _x = (-x).clip(min=0)
            _y = (-y).clip(min=0)
            _x1 = (x1-x).clip(max=target_w-x)
            _y1 = (y1-y).clip(max=target_h-y)
            # print(y,x,y1,x1, ',' ,_y,_x,_y1,_x1, resized.shape)
            resized = resized[:, _y:_y1, _x:_x1]
            w, h = _x1-_x, _y1-_y
            x = x.clip(min=0)
            y = y.clip(min=0)

        # q: [k,2]
        qi = 1.0 if q is None else 1.0 / q[i, 1]
        weight = 1.0
        if skip_win_for_0:
            skip_win_for_0 = False
        elif window is not None:
            # weight = TF.resize(window, [h,w], TF.InterpolationMode.BILINEAR, antialias=antialias)
            weight = F.interpolate(window, (h,w), mode='bilinear', align_corners=True)[0]

        # print(resized.shape, a_img[:,y:y1,x:x1].shape)
        a_img[:,y:y1,x:x1] += resized * (weight / qi)
        a_w[:,y:y1,x:x1] += weight

        if ret_resized:
            res.append(resized)

    a_img /= a_w.clamp_(min=1e-6)
    if ret_resized:
        return a_img, res
    return a_img


class EquiNorm(nn.Module):
    def __init__(self, layer, res, win, forward_mode, loss_mode, loss_fn=F.mse_loss):
        super().__init__()
        self.layer = layer
        # if in_dim is None:
        #     in_dim = layer.in_channels
        self.res = res
        self.win = win
        self.forward_mode = forward_mode
        self.loss_mode = loss_mode
        assert self.loss_mode <= 3
        self.loss_fn = loss_fn
        if args.predictor:
            self.pred = nn.Conv2d(layer.out_channels, layer.out_channels, kernel_size=1)
        else:
            self.pred = None

    def forward(self, f):
        # inputs: [n*k,in_c,h,w]
        # f: [n*k,c,h,w]
        f = self.layer(f)
        if self.forward_mode == 0 and self.loss_mode == 0:
            return f

        boxes, qs, masks = self.boxes
        n, k = qs.shape[:2]
        fa = []; resized = []
        ret_resized = self.loss_mode == 3
        for b in range(n):
            r = average(f[b*k:(b+1)*k], boxes[b], self.res, self.win,
                        q=qs[b] if args.equi_q else None, ret_resized=ret_resized)
            if ret_resized:
                fa.append(r[0])
                resized.append(r[1])
            else:
                fa.append(r)
        fa = torch.stack(fa)
        self.fa = fa

        feat_shape = f.shape[-2:]
        if self.loss_mode:
            with torch.no_grad():
                norm = (fa**2).mean()

        if self.loss_mode == 1:
            ft = F.interpolate(fa, feat_shape, mode='bilinear', align_corners=True)
            # ft = F.interpolate(fa, feat_shape, mode='area')
            fs = f[:n*k:k]
            self.loss = self.loss_fn(fs, ft, reduction='none').mean((1,2,3)) / norm

        elif self.loss_mode == 2:
            # forward mode ||f*t - t*fa||
            # for b in boxes:
            #     b[:,0].clamp_(min=0)
            #     b[:,1].clamp_(min=0)
            #     b[:,2].clamp_(max=self.res[1])
            #     b[:,3].clamp_(max=self.res[0])
            ft = fa
            if self.pred is not None:
                ft = ft.detach()
                ft = self.pred(ft)
            ft = torchvision.ops.roi_align(ft, boxes, feat_shape, aligned=True)
            fs = f[:n*k]
            # print((masks == -1).long().sum(), (masks == 0).long().sum(), (masks == 1).long().sum())
            masks = (masks == -1).expand_as(fs)
            fs[masks] = 0.  # prevent loss from propagating into the out portion
            self.loss = self.loss_fn(fs, ft, reduction='none').mean((1,2,3)) / norm

        elif self.loss_mode == 3:
            # backward mode ||t-1*f*t - fa||
            loss = []
            for b in range(n):
                l = torch.stack([
                    self.loss_fn(resized[b][_], fa[b, :, int(y):int(y1), int(x):int(x1)])
                    for _, (x,y,x1,y1) in enumerate(boxes[b])], dim=0)
                loss.append(l)
            loss = torch.cat(loss)
            self.loss = loss / norm

        if self.forward_mode == 0:
            return f
        if self.forward_mode == 1:
            return fa
        if self.forward_mode == 2:
            return torch.cat([f[:n*k], fa])
        if self.forward_mode == 3:
            return torch.cat([f, fa])
        # raise ValueError(f'unsupport forward mode: {self.forward_mode}')

    def extra_repr(self):
        return (
            f"res={self.res}, win={list(self.win.shape)}, forward_mode={self.forward_mode}, "
            f"loss_mode={self.loss_mode}, loss_fn={self.loss_fn.__module__}.{self.loss_fn.__name__}, "
            f"pred={self.pred}"
        )

def get_equi_layers(model):
    return [m for m in model.modules() if isinstance(m, EquiNorm)]

def set_equi_mode(equi_layers, forward_mode=None, loss_mode=None):
    for m in equi_layers:
        if forward_mode is not None:
            m.forward_mode = forward_mode
        if loss_mode is not None:
            m.loss_mode = loss_mode

def set_equi_boxes(equi_layers, boxes):
    for m in equi_layers:
        m.boxes = boxes


def get_equi_losses(equi_layers):
    return torch.stack([m.loss for m in equi_layers]).sum(dim=0)


def l1_loss(p, target, mask=None, reduction='mean'):
    if mask is not None:
        loss = F.l1_loss(p, target, reduction='none')
        mask = (mask > 0).float()
        if reduction == 'mean':
            loss = (loss * mask).sum() / mask.sum().clamp_(min=1e-6)
        elif reduction == 'batch':
            # todo: not exactly equivariant to 'mean'
            loss = (loss * mask).sum(dim=(1,2,3)) / mask.sum(dim=(1,2,3)).clamp_(min=1e-6)
        else:
            loss = loss * mask
    else:
        if reduction == 'mean':
            loss = F.l1_loss(p, target, reduction='mean')
        else:
            loss = F.l1_loss(p, target, reduction='none')
            if reduction == 'batch':
                loss = loss.mean(dim=(1,2,3))
    return loss


def l2_loss(p, target, mask=None, reduction='mean'):
    if mask is not None:
        loss = F.mse_loss(p, target, reduction='none')
        mask = (mask > 0).float()
        if reduction == 'mean':
            loss = (loss * mask).sum() / mask.sum().clamp_(min=1e-6)
        elif reduction == 'batch':
            # todo: not exactly equivariant to 'mean'
            loss = (loss * mask).sum(dim=(1,2,3)) / mask.sum(dim=(1,2,3)).clamp_(min=1e-6)
        else:
            loss = loss * mask
    else:
        if reduction == 'mean':
            loss = F.mse_loss(p, target, reduction='mean')
        else:
            loss = F.mse_loss(p, target, reduction='none')
            if reduction == 'batch':
                loss = loss.mean(dim=(1,2,3))
    return loss


def angular_err(p, target, mask=None):
    assert p.ndim >= 3
    # acos = (F.normalize(pred, dim=-3) * F.normalize(target.to(pred), dim=-3)
    #        ).sum(-3, keepdim=True).clip_(-1,1).acos_()
    acos = (F.normalize(p, dim=-3) * F.normalize(target.to(p), dim=-3)
           ).sum(-3, keepdim=True).clip_(-1,1).acos_()
    if mask is not None:
        mask = mask.to(loss)
        err = (acos * mask).sum() / mask.sum()
    else:
        err = acos.mean()
    return err * (180/np.pi)

# cross_entropy_loss = F.cross_entropy
def cross_entropy_loss(p, target, mask=None):
    if mask is None:
        # print(target.min(), target.max(), p.shape, target.shape)
        loss = F.cross_entropy(p, target.squeeze(1)-1, ignore_index=-1)
    else:
        loss = F.cross_entropy(p, target.squeeze(1)-1, ignore_index=-1, reduction='none')
        mask = mask.squeeze(1).to(loss)
        # print(p.shape, target.shape, loss.shape, mask.shape, target.min(), target.max(), loss.dtype, mask.dtype)
        loss = (loss * mask.squeeze(1)).sum() / mask.sum()
    return loss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, f='.3e'):
        self.reset()
        self.f = f

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{0:.3e} ({1:.3e})'.format(self.val, self.avg)


def iou(pred, target, n_classes):
    ious = []
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # Ignore IoU for target class < 0
    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    ious = np.array(ious)
    return ious, np.nanmean(ious)

def miou(p, target, mask=None):
    ious, miou = iou(p.argmax(1), target-1, out_dim)
    # print(ious, miou)
    return miou

def accuracy(p, target, mask=None):
    if mask is not None:
        acc = p.argmax(dim=1,keepdim=True).eq(target-1).float()
        mask = mask.to(acc)
        acc = (acc * mask).sum() / mask.sum()
    else:
        acc = p.argmax(dim=1,keepdim=True).eq(target-1).float().mean()
    return acc

# def update_q(losses, qs, q_net, q_lr=args.q_lr, q_reg=0.1):
#     qi = qs[:,1:,0].long()
#     losses = losses.detach().clone().view(*qs.shape[:2])[:,1:].to(q_net.device)
#     losses.div_(losses.mean())
#     q_net[qi] += q_lr * (losses + q_reg * (1.0 - q_net[qi]))
#     q_net.div_(q_net.mean())

def update_q(losses, qs, q_net, q_lr=args.q_lr, q_reg=1.0):
    qi = qs[:,1:,0].long()
    losses = losses.detach().clone().view(*qs.shape[:2])[:,1:].to(q_net.device)
    losses = -(losses - losses.mean()) / losses.std()
    # TODO: need to fix the duplicated index case
    q_net[qi] += q_lr * (losses + q_reg * (1.0 - q_net[qi]))
    q_net.div_(q_net.mean())


def fmt(m):
    return ', '.join(f'{k}: {v}' if isinstance(v, np.ndarray) else f'{k}: {v:.3e}' for k,v in m.items())

@torch.no_grad()
def val(model, dl, progress=True,
        # metrics={'l1': l1_loss, 'deg': angular_err},
        # metrics={'l1': l1_loss},
        metrics={'l1': l1_loss, 'l2': l2_loss},
        avg=True, epoch=None, reduction='batch', draw=False, tag='',
       ):
    assert reduction == 'batch'
    model.eval()
    # model.decoder.eval()
    equi_layers = get_equi_layers(model)
    set_equi_mode(equi_layers, forward_mode=2)
    transform = dl.transform

    num = 0
    if avg:
        md0 = {k:0 for k in metrics}
    md = {k:0 for k in metrics}
    md['eq'] = 0
    if progress:
        dl = tqdm(dl,desc='eval')

    for step, d in enumerate(dl):
        with torch.cuda.amp.autocast(enabled=bool(args.val_fp16)):
            d = transform(d, imp=False, color_aug=args.val_color_aug)

            rgbs, boxes, target, mask = d['rgb'], d['box'], d[task], d['mask']
            qs = d['q'].to(dev, non_blocking=True)
            n, k = qs.shape[:2]
            set_equi_boxes(equi_layers, (boxes, qs, mask))

            target = target[::k]
            # mask = d['mask'][::k] if has_mask else None
            mask = mask[::k]

            p = model(rgbs)
            if p.ndim == 3:
                p = p.unsqueeze(1) * 0.00024371305824304144
            p0 = p[:n*k:k]

        if draw and step % 500 == 0:
            i = 0
            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1); plt.imshow(pt2np(rgbs[i*k,:3],args.rgb_norm).astype(np.float32))
            if target.dtype is torch.int64:
                plt.subplot(1,3,2); plt.imshow(pt2np(target[i]),vmin=0,vmax=out_dim)
                plt.subplot(1,3,3); plt.imshow(pt2np(p[i].argmax(0)+1),vmin=0,vmax=out_dim)
            else:
                # plt.subplot(1,3,2); plt.imshow(pt2np(target[i]).astype(np.float32),vmin=0,vmax=1)
                # plt.subplot(1,3,3); plt.imshow(pt2np(p[i]).astype(np.float32),vmin=0,vmax=1)
                plt.subplot(1,3,2); plt.imshow(pt2np(1/target[i]), vmin=p[i].min().item(), vmax=p[i].max().item())
                plt.subplot(1,3,3); plt.imshow(pt2np(p[i]).astype(np.float32))
            plt.savefig(savedir+f'val_ep={epoch}_step={step}.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()

        # n = len(target)
        # n = mask.sum()
        for k, metric in metrics.items():
            if avg:
                md0[k] += metric(p0, target, mask, reduction='batch').mean(0).cpu().numpy() * n
            md[k] += metric(p[-n:], target, mask, reduction='batch').mean(0).cpu().numpy() * n
        md['eq'] += equi_layers[-1].loss.mean().item() * n
        num += n
        # if progress:
        #     dl.set_postfix({k: f'{md[k]/num:.3e}'})

    if progress:
        printf(dl, fileonly=True)
    for k in md:
        md[k] /= num
        if epoch is not None and args.tb:
            logger.add_scalar(f'loss/val_{k}_ep', md[k], epoch)
    if avg:
        for k in md0:
            md0[k] /= num
            if epoch is not None and args.tb:
                logger.add_scalar(f'loss/val_{k}_ep', md0[k], epoch)
        printf(f"loss: p0 {fmt(md0)} p {fmt(md)}")
        return md0, md
    printf(f"val loss: p {fmt(md)}")
    return md


def depth_loss(p, target, mask=None, reduction='mean'):
    return l1_loss(p, 1/target.clip_(min=1e-3), mask, reduction)


min_depth_eval = 0.001
# max_depth_eval = 1.
max_depth_eval = 10.

def compute_error_one_image(pred_disp, gt, align=True):
    if align:
        x = pred_disp; y = 1. / gt
        xm = x.mean(); x0 = x - xm
        ym = y.mean(); y0 = y - ym
        beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
        pred = x0 * beta + ym
    pred = (1 / pred.clip_(min=1./max_depth_eval)).clip_(min_depth_eval, max_depth_eval)
    gt = gt.clip(min_depth_eval, max_depth_eval)

    e = torch.maximum(gt / pred, pred / gt)
    d1 = (e < 1.25).float().mean()
    d2 = (e < 1.25 ** 2).float().mean()
    d3 = (e < 1.25 ** 3).float().mean()

    e = gt - pred
    rmse = (e ** 2).mean().sqrt()

    lg = gt.log()
    lp = pred.log()
    rmse_log = (lg - lp) ** 2
    rmse_log = rmse_log.mean().sqrt()

    abs_rel = (e.abs() / gt).mean()
    sq_rel = ((e ** 2) / gt).mean()

    e = lg - lp
    silog = torch.sqrt((e ** 2).mean() - e.mean() ** 2) * 100

    log10 = (lg - lp).abs().mean() / np.log(10)
    
    return torch.stack([silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3])


def depth_metrics(p, target, mask, reduction='batch'):
    # x = p; y = 1. / target
    # xm = x.mean(); x0 = x - xm
    # ym = y.mean(); y0 = y - ym
    # beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
    # pred = x0 * beta + ym
    # pred = (1 / pred).clip(min_depth_eval, max_depth_eval)
    mask = mask > 0
    # p = (1 / p).clip_(min_depth_eval, max_depth_eval)
    
    errors = [compute_error_one_image(p[i][mask[i]], target[i][mask[i]]) for i in range(len(mask))]
    errors = torch.stack(errors)
    if reduction == 'mean':
        errors = errors.mean(0)
    return errors


def train(model, dl, dl_val, dl_unl=None, num_epochs=1, equi=True, tag='', val0=True, vale=False):
    # load = False
    # _ = savedir+f'ep={num_epochs}.pt'
    # if os.path.exists(_):
    #     load = True
    # else:
    #     _ = savedir+f'{tag}_ep={num_epochs}.pt'
    #     if os.path.exists(_):
    #         load = True
    # if load:
    #     d = torch.load(_, map_location='cpu')['model'].state_dict()
    #     model.load_state_dict(d)
    #     printf(f'load weights from: {_}')
    #     return False
    # tagb = os.path.basename(tag)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(trainable_params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(dl))

    if task == 'segment_semantic':
        loss_fn = cross_entropy_loss
        metrics = {'ce': cross_entropy_loss, 'acc': accuracy}
    elif task == 'depth_zbuffer' or task == 'depth':
        loss_fn = depth_loss
        metrics = {'l': depth_loss, 'l1': l1_loss, 'm': depth_metrics}
    else:
        if not args.loss or args.loss == 'l1':
            loss_fn = l1_loss
        elif args.loss == 'l2':
            loss_fn = l2_loss
        metrics = {'l1': l1_loss, 'l2': l2_loss}
    printf(f"num train params: {len(optimizer.param_groups[0]['params'])}, loss_fn: {loss_fn.__name__}")
    printf(f'loss = {args.sup_coef} * loss_pred + {args.equi_coef} * loss_equi_lab + {args.a_coef} * loss_a + {args.unl_coef if unlabel else 0} * loss_equi_unl')

    if val0:
        printf(tag, end=' ')
        val(model, dl_val, metrics=metrics, epoch=0)
    t = time()

    scaler = torch.cuda.amp.GradScaler()
    equi_layers = get_equi_layers(model)
    transform = dl.transform

    unl = dl_unl is not None
    if unl:
        def infinite(dl):
            while True:
                for d in dl:
                    yield d
        it_unl = infinite(dl_unl)

    for epoch in range(0,num_epochs):
        model.train()
        set_equi_mode(equi_layers, forward_mode=0)

        a_loss = AverageMeter()
        a_1 = AverageMeter()
        a_loss_pred = AverageMeter()
        a_loss_equi = AverageMeter()
        a_loss_equi_unl = AverageMeter()
        a_loss_ave = AverageMeter()

        for step, d in enumerate(dl):
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=bool(args.train_fp16)):
                d = transform(d)

                n = len(d['q'])
                if unl:
                    du = next(it_unl)
                    du = transform(du)
                    n_unl = len(du['q'])
                    if not args.sep_batch:
                        for k in d:
                            if isinstance(d[k], torch.Tensor):
                                d[k] = torch.cat((d[k], du[k]))
                            elif isinstance(d[k], list):
                                d[k] = d[k] + du[k]
                            else:
                                raise ValueError(f'key {k} unsupported')

                # rgbs: (B*K,3+2,H,W), r,g,b + px,py
                # targets: (B*K,?,H,W)
                # boxes: [(K,4)]*B, i,j,h,w, qs: (B,K,2), iq,q
                rgbs, targets, boxes, qs, masks = d['rgb'], d[task], d['box'], d['q'], d.get('mask', None)

                n_all, k = qs.shape[:2]
                # n, k = batch_size, args.k
                set_equi_boxes(equi_layers, (boxes, qs, masks))

                p = model(rgbs)
                if p.ndim == 3:
                    p = p.unsqueeze(1) * 0.00024371305824304144

                if unl:
                    # nk1 = n*(k+1)
                    nk = n*k

                    loss_pred = loss_fn(p[:nk], targets[:nk], masks[:nk], reduction='batch').mean()

                    # if args.a_coef:
                    pa = equi_layers[-1].fa[:n]
                    loss_ave = loss_fn(pa, targets[:nk:k], masks[:nk:k], reduction='batch').mean()

                    if args.sep_batch:
                        # p_lab = p
                        # loss_pred = loss_fn(p_lab, targets, masks, reduction='batch').mean()
                        loss_equi_lab = get_equi_losses(equi_layers).mean()

                        d = du
                        rgbs, targets, boxes, qs, masks = d['rgb'], d[task], d['box'], d['q'], d.get('mask', None)
                        set_equi_boxes(equi_layers, (boxes, qs, masks))
                        p_unl = model(rgbs)

                        loss_equi_unl = get_equi_losses(equi_layers).mean()
                    else:
                        # n_unl = n_all - n
                        # p_lab, p_unl = p[:nk], p[nk:]

                        assert equi and not args.equi_q

                        loss_equi = get_equi_losses(equi_layers)
                        loss_equi_lab, loss_equi_unl = loss_equi[:nk].mean(), loss_equi[nk:].mean()

                    loss = loss_pred + args.equi_coef * loss_equi_lab + args.unl_coef * loss_equi_unl
                    if args.a_coef:
                        loss += args.a_coef * loss_ave

                else:
                    loss_pred = loss_fn(p, targets, masks, reduction='batch')
                    if equi:
                        loss_equi = get_equi_losses(equi_layers)
                        # loss_equi_lab, loss_equi_unl = loss_equi[:n_lab*k], loss_equi[n_lab*k:]
                        losses = args.sup_coef * loss_pred + args.equi_coef * loss_equi

                        # p, pa = p[:n*k], p[n*k:]
                        pa = equi_layers[-1].fa
                        loss_pred = loss_pred.mean()
                        loss_ave = loss_fn(pa, targets[::k], masks[::k], reduction='batch').mean()

                        loss_equi = loss_equi.mean()
                        if args.equi_q:
                            loss = (losses / qs[:,:,1].view(-1)).mean()
                        else:
                            loss = losses.mean()
                        if args.a_coef:
                            # + loss_equi * args.equi_coef, already computed in add_equi_loss
                            loss += args.a_coef * loss_ave
                    else:
                        loss = loss_pred = losses = loss_pred.mean()

                if args.equi_q:
                    update_q(losses, qs, transform.sampler.q)
                    if step % 20 == 0:
                        q = transform.sampler.q.view(transform.sampler.bins_scale, transform.sampler.bins_ratio,
                                                     transform.sampler.bins_xy, transform.sampler.bins_xy)
                        s = ''; names = ['s','r','y','x']
                        for _ in range(4):
                            dim = [0,1,2,3]; dim.remove(_)
                            s += names[_] + ': ' + ','.join('%.4f'%_ for _ in q.mean(dim).numpy()) + '  '
                        printf(s)

            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # n = len(rgbs)
            # n = mask.sum()
            a_loss.update(loss.item(), n)
            if task == 'normal':
                p = p.detach().to(torch.float32)
                err = angular_err(p, targets, masks)
                a_1.update(err.item(), n)
            elif task == 'segment_semantic':
                a = accuracy(p, targets, masks)
                a_1.update(a.item(), n)

            if equi:
                a_loss_pred.update(loss_pred.item(), n)
                a_loss_ave.update(loss_ave.item(), n)
                if unl:
                    a_loss_equi.update(loss_equi_lab.item(), n)
                    a_loss_equi_unl.update(loss_equi_unl.item(), n_unl)
                else:
                    a_loss_equi.update(loss_equi.item(), n)

            if step % args.print_freq == 0 or step == len(dl)-1:
                t1 = time()
                s = f"[ep {epoch:2d} step {step:4d}/{len(dl)} lr {lr:.1e} t {(time()-t)/60:.0f}m] loss: {a_loss}"
                if equi:
                    s += f" pred: {a_loss_pred}"
                    s += f" a: {a_loss_ave}"
                    s += f" equi: {a_loss_equi}"
                    if unl:
                        s += f" u {a_loss_equi_unl}"
                if task == 'normal':
                    s += f" err: {a_1}"
                elif task == 'segment_semantic':
                    s += f" acc: {a_1}"
                printf(s)

                if args.tb:
                    accu_step = epoch*len(dl)+step
                    logger.add_scalar('loss/train_loss_step', a_loss.val, accu_step)
                    logger.add_scalar('loss/train_pred_step', a_loss_pred.val, accu_step)
                    logger.add_scalar('loss/train_ave_step', a_loss_ave.val, accu_step)
                    logger.add_scalar('loss/train_equi_step', a_loss_equi.val, accu_step)
                    if unl:
                        logger.add_scalar('loss/train_equi_unl_step', a_loss_equi_unl.val, accu_step)

            if step % 1000 == 0 or step == len(dl)-1:
                m = 3
                plt.figure(figsize=(10,8))
                for i in range(3):
                    plt.subplot(3,m,1+i*m); plt.imshow(pt2np(rgbs[i,:3],args.rgb_norm).astype(np.float32))
                    if targets.dtype is torch.int64:
                        plt.subplot(3,m,2+i*m); plt.imshow(pt2np(targets[i]),vmin=0,vmax=out_dim)
                        plt.subplot(3,m,3+i*m); plt.imshow(pt2np(p[i].argmax(0)+1),vmin=0,vmax=out_dim)
                    else:
                        # plt.subplot(3,m,2+i*m); plt.imshow(pt2np(targets[i]).astype(np.float32),vmin=0,vmax=1)
                        plt.subplot(3,m,2+i*m); plt.imshow(pt2np(1/targets[i]), vmin=p[i].min().item(), vmax=p[i].max().item())
                        plt.subplot(3,m,3+i*m); plt.imshow(pt2np(p[i]).astype(np.float32))  # vmin=0,vmax=10
                    if m == 4:
                        plt.subplot(3,m,4+i*m); plt.imshow(pt2np(ws[i]).astype(np.float32),vmin=0,vmax=1)
                plt.savefig(savedir+f'ep={epoch}_step={step}.png', dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()

        if args.tb:
            logger.add_scalar('loss/train_loss_ep', a_loss.avg, epoch+1)
            logger.add_scalar('loss/train_pred_ep', a_loss_pred.avg, epoch+1)
            logger.add_scalar('loss/train_equi_ep', a_loss_equi.avg, epoch+1)
            if unl:
                logger.add_scalar('loss/train_equi_unl_ep', a_loss_equi_unl.avg, epoch+1)

        if (epoch != num_epochs-1 and (epoch+1) % args.val_freq == 0) or (epoch == num_epochs-1 and vale):
            printf(tag, end=' ')
            val(model, dl_val, metrics=metrics, epoch=epoch+1, draw=True, tag=tag)

        if (epoch+1) % args.save_freq == 0 or epoch == num_epochs-1:
            torch.save({'epoch': epoch+1, 'model': model.state_dict(), 'q': transform.sampler.q},
                       savedir+f'ep={epoch+1}.pt')
    return vale



to_tensor = transforms.ToTensor()
# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def transform(sample, size=(288,384)):
    sample['image'] = TF.resize(
        # norm(to_tensor(sample['image'].astype(np.float32))),
        to_tensor(sample['image'].astype(np.float32)),
        size,
        # transforms.InterpolationMode.BICUBIC,
        transforms.InterpolationMode.BILINEAR,
    )
    # sample['depth'] = to_tensor(sample['depth'].astype(np.float32))
    # sample['mask'] = to_tensor(sample['mask'].astype(np.float32))
    sample['depth'] = TF.resize(
        to_tensor(sample['depth'].astype(np.float32)),
        size, 
        # transforms.InterpolationMode.BILINEAR
        transforms.InterpolationMode.NEAREST,
    )
    sample['mask'] = TF.resize(
        to_tensor(sample['mask'].astype(np.float32)),
        size, transforms.InterpolationMode.NEAREST)
    return sample


batch_size = 8
num_workers = 10

ds_train = NyuDepthV2(transform=transform, split='train')
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True
                     )

ds_test = NyuDepthV2(transform=transform, split='test')
dl_test = DataLoader(ds_test, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True
                    )

args.rgb_norm = True
augk = DictAug(k=args.k,
              box_type=args.box_type
              # box_type='s0.6-1.1,op0.1'
             )
aug1 = DictAug(k=1, box_type='s1-1,r,op0')
dl_test.transform = dl_train.transform = augk
printf(augk)


unlabel = 0
args.lr = 1e-5
args.print_freq = 100
args.val_freq = 1

printf('\n\n','>'*80, '\n', 'color_aug =', args.color_aug, 'box_type =', args.box_type)

for args.epochs in [10]:
    # printf('='*80, '\n', 'ep =', args.equi_coef)
    for args.equi_coef in [0.00001]:
        if args.equi_coef is None:
            # if args.epochs == 5: continue
            # args.equi_coef = 0
            args.k = 1
            dl_train.transform = aug1
        else:
            args.k = 3
            dl_train.transform = augk

        printf('\n\n','='*80, '\n', 'equi_coef =', args.equi_coef, 'ep =', args.epochs, 'k =', args.k)
        if args.equi_coef is None:
            args.equi_coef = 0

        # args.predictor = False
        if args.net == 'midas_v21':
            model = MidasNet(model_path, non_negative=True)
        elif args.net == 'dpt_large':
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )

        loss_mode = {'forward': 2, 'backward': 3}[args.equi_loss_mode]
        for _ in range(len(model.scratch.output_conv)-1,-1,-1):
            if isinstance(model.scratch.output_conv[_], nn.Conv2d):
                printf(f'find last conv: {_}')
                model.scratch.output_conv[_] = EquiNorm(
                    # model.scratch.output_conv.layer,
                    model.scratch.output_conv[_],
                    res=(288,384), win=win, forward_mode=0, loss_mode=loss_mode)
                break

        model = model.to(dev)
        if args.equi_coef == 0: printf(model.scratch.output_conv)

        # with torch.autograd.set_detect_anomaly(True):
        train(
            model, dl_train, dl_test, None,
            num_epochs=args.epochs, equi=True, tag=args.tag,
            # val0=args.val0
            val0=1
        )

        metrics = {'l': depth_loss, 'l1': l1_loss, 'm': depth_metrics}
        val(model, dl_test, metrics=metrics)

