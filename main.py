import sys
import os
import numpy as np
from tqdm import tqdm
from time import time
import builtins

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms, utils, models
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from matplotlib import patches

# import sys
# sys.path.insert(0,os.path.expandvars('$WORK/omnidata'))
# from omnidata_tools.torch.data import taskonomy_dataset, splits
from omnidata import taskonomy_dataset, splits


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=5, linewidth=100)

    mean_=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1); std_=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)  # cifar10
    # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)  # imagenet
    # img_size = args.img_size

    last_act = None
    if args.task == 'edge_texture':
        has_mask = False; out_dim = 1
    elif args.task == 'keypoints3d':
        has_mask = False; out_dim = 1
    elif args.task == 'segment_semantic':
        has_mask = True; out_dim = 17; last_act = F.log_softmax
        # last_act = lambda x: F.log_softmax(x, dim=1)
    elif args.task == 'normal':
        has_mask = True; out_dim = 3
    elif args.task == 'depth_zbuffer':
        has_mask = True; out_dim = 1
    else:
        raise ValueError(f'unsupported task: {args.task}')
    num_workers = 10
    num_bldg = args.num_bldg


    # if args.tag.endswith('.pt'):
    #     args.tag = os.path.dirname(args.tag)
    args.savedir = f'{args.dir}/{args.task}_{args.tag}/'
    os.makedirs(args.savedir, exist_ok=True)
    if args.local_rank == 0:
        file = open(args.savedir+f'_log.txt', 'a')
        print_ = print
        def myprint(*args, fileonly=False, **kwargs):
            if not fileonly:
                print_(*args, **kwargs)
            print_(*args, **kwargs, file=file)
            file.flush()
        myprint.print_ = print_
        builtins.print = myprint

        # if args.verbose:
        print(args)
        if args.tb:
            logger = SummaryWriter(args.savedir)
    else:
        args.tb = False


    dl_trains = {}
    dl_vals = {}
    dl_unls = {}
    # norm = transforms.Normalize(mean=mean_.view(-1), std=std_.view(-1))
    # unlabel = args.percent and args.percent != 1
    # unlabel = bool(args.unlabel)

    ks = [1,3]
    if args.k not in ks:
        ks = sorted(ks + [args.k])
    for k in ks:
        my_transform = DictAug((args.img_size, args.img_size),
                               k=k,
                               pos_encode=args.pos_encode,
                               box_type=args.box_type,
                               verbose=args.local_rank==0)

        options = taskonomy_dataset.TaskonomyDataLoader.Options()
        # options.data_path='/home/x-yyz/proj/taskonomy_dataset'
        options.data_path='/tmp/yuanyi/taskonomy_dataset'
        # options.data_path='../taskonomy_dataset'
        options.bin_root=options.data_path
        options.tasks=['rgb', args.task]
        # options.buildings='debug-train'
        # options.buildings='tiny-train'
        options.buildings=splits.taskonomy_split_to_buildings['tiny']['train'][:num_bldg]
        if args.verbose and args.local_rank == 0:
            print(options.buildings)
        options.return_mask=has_mask
        options.phase='train'
        options.batch_size=args.batch_size
        options.shuffle=not args.distributed
        options.image_size=args.img_size
        options.num_workers=num_workers
        # options.transform=train_transform  # joint transform for both input and target
        # print(options)

        ds_train = taskonomy_dataset.TaskonomyDataset(options, verbose=args.verbose and args.local_rank == 0)
        # dl_train = taskonomy_dataset.TaskonomyDataLoader.make(options)
        if args.img_size == 256:
            # remove the Resize(img_size)
            del ds_train.transform[0].transforms[0]
            ds_train.transform[1] = ds_train.transform[1].transforms[1]
        # transform = ds_train.transform
        # ds_train.transform[0].transforms.append(norm)
        ds_transform = ds_train.transform

        if args.label:
            num_labeled = args.label
            if args.local_rank == 0: print(f'lab: {len(ds_train)} -> {num_labeled} ({num_labeled/len(ds_train)*100:.1f}%)')
            ds_train, ds_unl = torch.utils.data.random_split(
                ds_train, [num_labeled, len(ds_train) - num_labeled],
                generator=torch.Generator().manual_seed(42))
            ds_train.transform = ds_transform

        ### TODO
        if args.unlabel:
            assert args.label
            # num_labeled = len(ds_train)
            num_unlabeled = args.unlabel

            # options.buildings=splits.taskonomy_split_to_buildings['tiny']['train'][num_bldg:]
            # ds_unl = taskonomy_dataset.TaskonomyDataset(options)
            # ds_unl = ds_train
            if args.local_rank == 0: print(f'unl: {len(ds_unl)} -> {num_unlabeled} ({num_unlabeled/len(ds_unl)*100:.1f}%)')
            ds_unl, _ = torch.utils.data.random_split(
                ds_unl, [num_unlabeled, len(ds_unl) - num_unlabeled],
                generator=torch.Generator().manual_seed(42))

            # following seems to be safe -- each time i get the same ds_train regardless of the split sizes; but anyway we choose safer way
            # ds_train, ds_unl, _ = torch.utils.data.random_split(ds_unl, [num_labeled, num_unlabeled, len(ds_unl) - num_unlabeled - num_labeled],
            #                                           generator=torch.Generator().manual_seed(42))

            # ds_train.transform = ds_unl.transform = ds_transform
            ds_unl.transform = ds_transform

            if args.distributed:
                sampler = DistributedSampler(ds_unl, shuffle=True)  # seed=0, drop_last=True
            else:
                sampler = None
            dl_unl = taskonomy_dataset.TaskonomyDataLoader.make(options, dataset=ds_unl, sampler=sampler)

            dl_unls[f'{k}'] = dl_unl
            if args.local_rank == 0:
                print(f'unlabeled: bldgs={len(options.buildings)}, percent={num_unlabeled}/{num_labeled}={num_unlabeled/num_labeled:.3f}:',
                      options.buildings, 'k={k}, batches=len(dl_unl)}, images=len(dl_unl.dataset)')

            dl_unl.transform = my_transform

        if args.distributed:
            sampler = DistributedSampler(ds_train, shuffle=True)  # seed=0, drop_last=True
        else:
            sampler = None
        dl_train = taskonomy_dataset.TaskonomyDataLoader.make(options, dataset=ds_train, sampler=sampler)

        dl_trains[f'{k}'] = dl_train
        if args.local_rank == 0: print(f'train: k={k}, batches={len(dl_train)}, images={len(dl_train.dataset)}, sampler={sampler}')

        # options.buildings='debug-val'
        if args.val == 'tiny':
            options.buildings='tiny-val'
        elif args.val == 'c':
            options.buildings=['collierville']  # 3285 imgs, 206 batches
        else:
            raise ValueError(f'unknown {args.val}')
        # options.buildings=['noxapater']  # 4436 imgs, 278 batches
        options.phase='val'
        # options.batch_size=1
        options.shuffle=False
        # options.load_to_mem=True

        ds_val = taskonomy_dataset.TaskonomyDataset(options, verbose=args.verbose and args.local_rank == 0)
        if args.distributed:
            sampler = DistributedSampler(ds_val, shuffle=False)  # seed=0, drop_last=True
        else:
            sampler = None
        dl_val = taskonomy_dataset.TaskonomyDataLoader.make(options, ds_val, sampler=sampler)
        dl_val.dataset.transform = dl_train.dataset.transform

        dl_vals[f'{k}'] = dl_val
        if args.local_rank == 0: print(f'val: k={k}, batches={len(dl_val)}, images={len(dl_val.dataset)}, sampler={sampler}')

        dl_train.transform = dl_val.transform = my_transform

    # unlabel = unlabel and args.unl_batch

    if args.verbose and args.local_rank == 0:
        print(dl_train.dataset.transform, dl_val.dataset.transform, dl_unl.dataset.transform if args.unlabel else None)
        print(dl_train.transform, dl_val.transform, dl_unl.transform if args.unlabel else None, '\n')
        # print(my_transform, '\n')

    # win = cos_window((32,32), 5)
    win = cos_window((128,128), 20).to(dev)
    win = win.reshape(1,1,*win.shape[-2:])
    # plt.imshow(win); plt.show()


    if args.local_rank == 0: print('\n'+args.task)
    if args.pos_encode:
        in_dim = 5
    else:
        in_dim = 3
    loss_mode = {'forward': 2, 'backward': 3}[args.equi_loss_mode]
    if args.net == 'simple':
        model = nn.Sequential(
            nn.Conv2d(in_dim, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, 5, padding=2),
            EquiNorm(res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode),
            # EquiNorm(
            #     nn.Conv2d(32, out_dim, 5, padding=2),
            #     res=(img_size,img_size), win=win, forward_mode=0, loss_mode=loss_mode, w_mode=args.equi_w),
        )
    elif args.net.startswith('unet'):
        # import sys
        # sys.path.insert(0,os.path.expandvars('../XTConsistency'))
        # from modules.unet import UNet
        from unet import UNet

        if args.net != 'unet':
            ds = int(args.net[4:])
        else:
            ds = 6  # default 6 levels of downsample
        model = UNet(downsample=ds, in_channels=in_dim, out_channels=out_dim, last_act=last_act)

        dim = out_dim
        for l in args.equi_layers.split(','):
            # if l == 'up_blocks[0]':
            if l[:2] == 'up':
                block = int(l[2])
                loc = l[3:]
                if block == 0:
                    dim = 48 if loc == 'cat' else 16
                elif block == 1:
                    dim = 96 if loc == 'cat' else 32
                elif block == 2:
                    dim = 192 if loc == 'cat' else 64
                elif block == 3:
                    dim = 384 if loc == 'cat' else 128
                elif block == 4:
                    dim = 768 if loc == 'cat' else 256
                if loc == 'end':
                    model.up_blocks[block].bn3 = EquiNorm(
                        model.up_blocks[block].bn3,
                        res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=dim,
                        before=False)
                elif loc == 'c1':
                    model.up_blocks[block].conv1 = EquiNorm(
                        model.up_blocks[block].conv1,
                        res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=dim,
                        before=False)
                elif loc == 'cat':
                    model.up_blocks[block].conv1 = EquiNorm(
                        model.up_blocks[block].conv1,
                        res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=dim,
                        before=True)
            # if l[-1] == ']':
            #     # set an array element
            #     _ = int(l[l.rfind('[')+1:l.rfind(']')])
            #     layer = getattr(model, l[:l.rfind('[')])
            #     layer[_] = EquiNorm(
            #         layer[_],
            #         res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=dim)
            else:
                setattr(model, l, EquiNorm(
                    getattr(model, l),
                    res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=dim))
    elif args.net.startswith('resnet'):
        assert last_act is None
        assert in_dim == 3

        from resnet_depth import ResNetDepth
        pretrained = False
        if args.net != 'resnet':
            v = args.net.split('-')[1] # variant
            if v == 'pt':
                pretrained = True
            else:
                pretrained = v
                # raise ValueError(f'unknown resnet variant: {v}')

        model = ResNetDepth(out_dim=out_dim, pretrained=pretrained)

        for l in args.equi_layers.split(','):
            if l == 'minus0':
                model.decoder = EquiNorm(
                    model.decoder,
                    res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=out_dim,
                    before=False)
            elif l.startswith('minus'):
                l = -int(l[-1])-1
                assert l >= -5
                model.decoder[l] = EquiNorm(
                    model.decoder[l],
                    res=(args.img_size,args.img_size), win=win, forward_mode=0, loss_mode=loss_mode, dim=128,
                    before=False)
            else:
                raise ValueError(f'unsupported equi_layers: {l}')
    else:
        raise ValueError(f'unsupported net: {args.net}')
    model = model.to(dev)

    # only need to do it once at rank=0; get broadcast naturally
    if args.init and args.local_rank == 0:
        print(f'init from: {args.init}')
        print(model.load_state_dict(torch.load(args.init, map_location='cpu')['model'].state_dict(), strict=False))

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev], output_device=dev)

    # if args.verbose and args.local_rank == 0:
    if args.local_rank == 0:
        print(model)


    if args.task == 'normal':
        loss_fn = l1_loss
        metrics = {'l1': l1_loss, 'l2': l2_loss, 'deg': angular_err}
    elif args.task == 'segment_semantic':
        loss_fn = cross_entropy_loss
        metrics = {'ce': cross_entropy_loss, 'miou_macc_aacc': iou}
        # 'acc': accuracy, 'iou': iou
    elif args.task == 'depth_zbuffer':
        loss_fn = depth_loss
        # metrics = {'l': depth_loss, 'l1': l1_loss, 'm': depth_metrics}
        metrics = {'l': depth_loss, 'm': depth_metrics}
    else:
        if not args.loss or args.loss == 'l1':
            loss_fn = l1_loss
        elif args.loss == 'l2':
            loss_fn = l2_loss
        metrics = {'l1': l1_loss, 'l2': l2_loss}

    # training
    k = f'{args.k}'
    if args.test_only:
        kv = None
        _ = False
    else:
        kv = '3' if int(k) <= 1 else k
        _ = train(model, dl_trains[k], dl_vals[kv], dl_unls[k] if args.unlabel else None,
                  num_epochs=args.epochs, equi=True, tag=args.tag, val0=args.val0, metrics=metrics, loss_fn=loss_fn)

    for kk in dl_trains:
        if not _ or kk != kv:
            if args.local_rank == 0: print(f'val{kk}')
            val(model, dl_vals[kk], metrics=metrics);
    if args.local_rank == 0: print(f'train{k}')
    val(model, dl_trains[k], metrics=metrics);
    if k != '3':
        if args.local_rank == 0: print(f'train{3}')
        val(model, dl_trains['3'], metrics=metrics);


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


# NOTE: this function taskes list of xywh, while the pytorch roi_align takes xyxy!
def my_roi_align(input, boxes, output_size, interpolation=InterpolationMode.BILINEAR):
    crops = []
    for im, b in zip(input, boxes):
        # for bb in b.round().int():
        for bb in b:
            x,y,w,h = bb
            t = TF.resized_crop(im, y, x, h, w, output_size, interpolation=interpolation)
            # x0,y0,x1,y1 = bb
            # t = TF.resized_crop(im, y0, x0, y1-y0, x1-x0, output_size, interpolation=interpolation)
            crops.append(t)
    return torch.stack(crops)


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
            print(f'load_q from: {args.load_q}')
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
    def __init__(self, size, k=3, pos_encode=0, box_type='s0.4-1,r,op0.2', verbose=False,
                 image_interp=InterpolationMode.BILINEAR, target_interp=InterpolationMode.NEAREST):
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
        self.default_mask = torch.full((1,1,h,w), 2.0, device=args.dev)

        if args.color_aug:
            strength = args.color_strength
            self.color_aug = transforms.ColorJitter(
                brightness=strength,
                contrast=strength,
                saturation=strength,
                hue=args.hue_ratio*strength)
            if verbose: print(f'use color aug: [{strength},{strength},{strength},{args.hue_ratio*strength}]')
        if args.rgb_norm:
            self.norm = transforms.Normalize(mean=mean_.view(-1), std=std_.view(-1))
            if verbose: print(f'use rgb norm: {self.norm}')
        else:
            self.norm = lambda x: x

        self.image_interp = image_interp
        self.target_interp = target_interp

    def __call__(self, d, color_aug, imp=True):
        dev = args.dev
        # (B,3,H,W), (B,?,H,W), (B,1,H,W)
        # task = 'depth'
        if 'image' in d:
            rgbs = d['image']
        else:
            rgbs = d['rgb']
        rgbs = rgbs.to(dev, non_blocking=True)
        targets = d[args.task].to(dev, non_blocking=True)
        # rgb = rgb.view(-1, *rgb.shape[2:])
        n = len(rgbs)
        if 'mask' in d:
            masks = d['mask'].to(dev, non_blocking=True) + 1
        else:
            masks = self.default_mask.expand(n,-1,-1,-1)

        assert isinstance(rgbs, torch.Tensor)
        h0, w0 = rgbs.shape[-2:]  # PIL image.size is (width, height), pytorch.shape is (height, width)

        # randomly make k crops for each image
        k = max(1, self.k)
        list_boxes = []; qs = torch.empty((n,k,2), dtype=torch.float32, device=dev)
        for b in range(n):
            boxes = []
            for _ in range(0,k):
                if _ == 0 and self.k != 1:
                    box = (0, 0, self.size[1], self.size[0])  # size=(h,w) -> (w,h)
                    qs[b,0,0], qs[b,0,1] = -1, 1.
                else:
                    i, j, h, w, qs[b,_,0], qs[b,_,1] = self.sampler(*self.size, imp=imp)
                    # box = j, i, j+w, i+h  # roi_align takes x0,y0,x1,y1 format, but interpolate takes y,x,h,w
                    box = j, i, w, h
                boxes.append(box)
            # list_boxes.append(torch.tensor(boxes, dtype=torch.float32, device=dev))
            # list_boxes.append(torch.tensor(boxes, dtype=torch.int32, device=dev))
            list_boxes.append(boxes)

        if not color_aug and args.rgb_norm:
            rgbs = self.norm(rgbs)

        if self.pos_encode:
            rgbs = torch.cat([rgbs, self.pos[None].expand(n,-1,-1,-1)], dim=1)
        
        rgbs = my_roi_align(rgbs, list_boxes, self.size, self.image_interp)
        targets = my_roi_align(targets, list_boxes, self.size, self.target_interp)
        if masks is not None:
            # NEAREST to avoid issues, automatically pad 0 for outside regions
            masks = my_roi_align(masks, list_boxes, self.size, InterpolationMode.NEAREST)

        # crops = my_roi_align(stack, list_boxes, self.size, aligned=True, sampling_ratio=1)
        if color_aug:
            for _ in  range(len(rgbs)):
                if color_aug == 2 and _ % k == 0:
                    if args.rgb_norm:
                        rgbs[_,:3] = self.norm(rgbs[_,:3])
                else:  # color_aug == 1
                    rgbs[_,:3] = self.norm(self.color_aug(rgbs[_,:3]))

        r = {'rgb': rgbs, args.task: targets, 'box': list_boxes, 'q': qs,
            # 'orig_depth': targets, 'orig_mask': masks
            }
        if masks is not None:
            r['mask'] = masks - 1
            # mask -1: padding due to cropping, 0: originally invalid pixels, 1: valid pixels
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


def lstsq_align(x, y):
    s = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1)
    xm = x.mean(); x0 = x - xm
    ym = y.mean(); y0 = y - ym
    beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
    p = x0 * beta + ym
    return p.reshape(s)


def average(crops, boxes, target_size, window=None, skip_win_for_0=True,
            q=None, ret_resized=False,
            # lstsq=False
           ):
    # boxes = boxes.int().cpu().numpy()  # will this be faster than gpu?

    n = len(crops)
    dev = crops.device
    a_img = torch.zeros((crops.shape[1],) + target_size, device=dev)
    a_w = torch.zeros((1,) + target_size, device=dev)
    target_h, target_w = target_size

    # if window is not None:
    #     window = window.view(1,1,*window.shape[-2:])

    res = []
    for i, crop, xywh in zip(range(n), crops, boxes):
        x,y,w,h = xywh
        x1,y1 = x+w,y+h
        # x,y,x1,y1 = xyxy
        # w, h = x1-x, y1-y
        resized = F.interpolate(crop[None], (h,w), mode='bilinear', align_corners=True)[0]
        if x < 0 or y < 0 or x1 > target_w or y1 > target_h:
            _x = max(-x, 0)
            _y = max(-y, 0)
            _x1 = min(x1-x, target_w-x)
            _y1 = min(y1-y, target_h-y)
            resized = resized[:, _y:_y1, _x:_x1]
            w, h = _x1-_x, _y1-_y
            x = max(x, 0)
            y = max(y, 0)
            # _x = (-x).clip(min=0)
            # _y = (-y).clip(min=0)
            # _x1 = (x1-x).clip(max=target_w-x)
            # _y1 = (y1-y).clip(max=target_h-y)
            # # print(y,x,y1,x1, ',' ,_y,_x,_y1,_x1, resized.shape)
            # resized = resized[:, _y:_y1, _x:_x1]
            # w, h = _x1-_x, _y1-_y
            # x = x.clip(min=0)
            # y = y.clip(min=0)

        # q: [k,2]
        qi = 1.0 if q is None else 1.0 / q[i, 1]
        weight = 1.0
        if skip_win_for_0:
            skip_win_for_0 = False
        elif window is not None:
            # weight = TF.resize(window, [h,w], InterpolationMode.BILINEAR, antialias=antialias)
            weight = F.interpolate(window, (h,w), mode='bilinear', align_corners=True)[0]

        # print(resized.shape, a_img[:,y:y1,x:x1].shape)
        # if lstsq and i > 0:
        #     with torch.no_grad():
        #         resized = lstsq_align(resized, a_img[:,y:y1,x:x1])
        a_img[:,y:y1,x:x1] += resized * (weight / qi)
        a_w[:,y:y1,x:x1] += weight

        if ret_resized:
            res.append(resized)

    a_img /= a_w.clamp_(min=1e-6)
    if ret_resized:
        return a_img, res
    return a_img


class EquiNorm(nn.Module):
    def __init__(self, layer, res, win, forward_mode, loss_mode, loss_fn=F.mse_loss, dim=None, before=False):
        super().__init__()
        self.layer = layer
        self.res = res
        self.win = win
        self.forward_mode = forward_mode
        self.loss_mode = loss_mode
        assert self.loss_mode <= 3
        self.loss_fn = loss_fn

        if args.predictor:
            if before:
                dim = layer.in_channels if hasattr(layer, 'in_channels') else dim
            else:
                dim = layer.out_channels if hasattr(layer, 'out_channels') else dim
            assert dim is not None, 'must specify the feature dimension for predictor'
            # no bias to avoid problems
            self.pred = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            self.pred.weight.data.copy_(torch.eye(dim)[:,:,None,None])

            # self.pred = nn.Conv2d(dim, dim, kernel_size=1)
            # self.pred.weight.data.copy_(torch.eye(dim)[:,:,None,None])
            # self.pred.bias.data.mul_(0.0)
        else:
            self.pred = None
        self.before = before  # do equinorm before f or after f
        self.warn = 0

    def forward(self, f):
        # inputs: [n*k,in_c,h,w]
        # f: [n*k,c,h,w]
        if self.forward_mode == 0 and self.loss_mode == 0:
            return self.layer(f)
        if not self.before:
            f = self.layer(f)

        boxes, qs, masks = self.boxes
        n, k = qs.shape[:2]
        fa = []; resized = []
        ret_resized = self.loss_mode == 3

        for b in range(n):
            r = average(f[b*k:(b+1)*k], boxes[b], self.res, self.win,
                        q=qs[b] if args.equi_q else None, ret_resized=ret_resized,
                        # lstsq=args.lstsq
                       )
            if ret_resized:
                fa.append(r[0])
                resized.append(r[1])
            else:
                fa.append(r)
        fa = torch.stack(fa)
        # self.fa = fa

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
                # ft = self.pred(ft)
                # print(self.pred.weight, self.pred.bias)
            # ft = torchvision.ops.roi_align(ft, boxes, feat_shape, aligned=True)
            ft = my_roi_align(ft, boxes, feat_shape)
            fs = f[:n*k]
            if self.pred is not None:
                fs = self.pred(fs)
            # print((masks == -1).long().sum(), (masks == 0).long().sum(), (masks == 1).long().sum())
            # masks = (masks == -1).expand_as(fs)
            # fs[masks] = 0.  # prevent loss from propagating into the out portion
            # masks = (masks != -1).expand_as(fs).float()  # [8,17,256,256]
            # fs = fs * masks

            # for mid_conv3, this could be problematic:
            # torch.Size([24, 1024, 4, 4]) torch.Size([24, 1024, 4, 4]) torch.Size([24, 1, 256, 256])
            # print(fs.shape, ft.shape, masks.shape)
            masks = (masks != -1).float()  # [8,1,256,256], exclude the outside regions
            if masks.shape[-1] != feat_shape[-1]:
                if not (self.warn & 1) and args.local_rank == 0:
                    print(f'masks shape mismatch, resize from {masks.shape} to {fs.shape}')
                    self.warn |= 1
                masks = F.interpolate(masks, feat_shape, mode='bilinear', align_corners=True)

            self.loss = (self.loss_fn(fs, ft, reduction='none') * masks).mean((1,2,3)) / norm
            # self.loss = self.loss_fn(fs, ft, reduction='none').mean((1,2,3)) / norm

        elif self.loss_mode == 3:
            raise ValueError('unsupported loss_mode 3')
            # backward mode ||t-1*f*t - fa||
            # loss = []
            # for b in range(n):
            #     l = torch.stack([
            #         self.loss_fn(resized[b][_], fa[b, :, int(y):int(y1), int(x):int(x1)])
            #         for _, (x,y,x1,y1) in enumerate(boxes[b])], dim=0)
            #     loss.append(l)
            # loss = torch.cat(loss)
            # self.loss = loss / norm

        if fa.shape[-1] != feat_shape[-1]:
            if not (self.warn & 2) and args.local_rank == 0:
                print(f'fa shape mismatch, resize from {fa.shape} to {f.shape}')
                self.warn |= 2
            fa = F.interpolate(fa, feat_shape, mode='bilinear', align_corners=True)
        # if self.forward_mode == 0:
        #     f = f
        if self.forward_mode == 1:
            f = fa
        elif self.forward_mode == 2:
            f = torch.cat([f[:n*k], fa])
        elif self.forward_mode == 3:
            f = torch.cat([f, fa])
        # raise ValueError(f'unsupport forward mode: {self.forward_mode}')
        if self.before:
            return self.layer(f)
        return f

    def extra_repr(self):
        return (
            f"res={self.res}, win={list(self.win.shape)}, forward_mode={self.forward_mode}, "
            f"loss_mode={self.loss_mode}, loss_fn={self.loss_fn.__module__}.{self.loss_fn.__name__}, "
            f"pred={self.pred}, before={self.before}"
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


def angular_err(p, target, mask=None, reduction='batch'):
    # assert p.ndim >= 3
    # arccos
    loss = (F.normalize(p, dim=-3) * F.normalize(target.to(p), dim=-3)
           ).sum(-3, keepdim=True).clip_(-1,1).acos_()
    if mask is not None:
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
            loss = loss.mean()
        else:
            if reduction == 'batch':
                loss = loss.mean(dim=(1,2,3))
    return loss * (180/np.pi)


# NOTE: uses F.nll_loss rather than F.cross_entropy. assumes input is after log_softmax
def cross_entropy_loss(p, target, mask=None, reduction='batch'):
    target = target.long().squeeze(1)-1
    # print(p.shape, target.shape, target.dtype, mask.shape)  # target: [b,1,h,w], float32
    if mask is None:
        # print(target.min(), target.max(), p.shape, target.shape)
        # loss = F.cross_entropy(p, target, ignore_index=-1)
        loss = F.nll_loss(p, target, ignore_index=-1)
    else:
        # loss = F.cross_entropy(p, target, ignore_index=-1, reduction='none')
        loss = F.nll_loss(p, target, ignore_index=-1, reduction='none')
        # mask = mask.squeeze(1).to(loss)
        mask = (mask.squeeze(1) > 0).float()
        # print(p.shape, target.shape, loss.shape, mask.shape, target.min(), target.max(), loss.dtype, mask.dtype)
        if reduction == 'mean':
            loss = (loss * mask).sum() / mask.sum()
        elif reduction == 'batch':
            # todo: not exactly equivariant to 'mean'
            loss = (loss * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_(min=1e-6)
    return loss


def count(x, n_classes):
    c = x.reshape(-1).bincount().float()
    if len(c) < n_classes:
        c = torch.cat([c, torch.zeros(n_classes - len(c), dtype=c.dtype, device=c.device)])
    return c

def iou(p, t, mask=None, reduction=None, num_ignore=1):
    n = p.shape[0]
    n_classes = p.shape[1]
    p = p.argmax(1).reshape(-1)
    t = t.reshape(-1).to(p) - num_ignore
    # print(p.shape, t.shape, mask.shape, p.min(), p.max(), t.min(), t.max())
    if mask is not None:
        mask = (mask > 0).reshape(-1)
        p = p[mask]
        t = t[mask]
    intersection = count(t[p.eq(t)], n_classes)
    target = count(t, n_classes)
    union = count(p, n_classes) + target - intersection
    return torch.stack([intersection, target, union])[None] / n  # hack to deal with val code

def compute_miou_macc_aacc_from_statistics(stats):
    intersection, target, union = stats.reshape(3, -1)
    miou = (intersection / union).nanmean()
    macc = (intersection / target).nanmean()
    aacc = intersection.sum() / target.sum()
    return torch.stack([miou, macc, aacc])

def miou_macc_aacc(p, t, mask=None):
    return compute_miou_macc_aacc_from_statistics(iou(p, t, mask))



def depth_loss(p, target, mask=None, reduction='mean'):
    return l1_loss(p, 1/target.clip_(min=1e-3), mask, reduction)


def compute_error_one_image(pred, gt):
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
    min_depth_eval = 0.001
    max_depth_eval = 1.
    
    # x = p; y = 1. / target
    # xm = x.mean(); x0 = x - xm
    # ym = y.mean(); y0 = y - ym
    # beta = (x0[...,None,:] @ y0[...,None])[...,0,0] / (x0[...,None,:] @ x0[...,None])[...,0,0]
    # pred = x0 * beta + ym
    # pred = (1 / pred).clip(min_depth_eval, max_depth_eval)
    mask = mask > 0
    p = (1 / p).clip_(min_depth_eval, max_depth_eval)
    
    errors = [compute_error_one_image(p[i][mask[i]], target[i][mask[i]]) for i in range(len(mask))]
    errors = torch.stack(errors)
    if reduction == 'mean':
        errors = errors.mean(0)
    return errors


def tstr(tensor):
    s = str(tensor)
    return s[s.find('['):s.rfind(']')+1]

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
        if isinstance(self.val, torch.Tensor):
            return '{0} ({1})'.format(tstr(self.val), tstr(self.avg))
        return '{0:.3e} ({1:.3e})'.format(self.val, self.avg)


# def update_q(losses, qs, q_net, q_lr=args.q_lr, q_reg=1.0):
#     qi = qs[:,1:,0].long()
#     losses = losses.detach().clone().view(*qs.shape[:2])[:,1:].to(q_net.device)
#     losses = -(losses - losses.mean()) / losses.std()
#     # TODO: need to fix the duplicated index case
#     q_net[qi] += q_lr * (losses + q_reg * (1.0 - q_net[qi]))
#     q_net.div_(q_net.mean())


def fmt(m):
    r = []
    for k,v in m.items():
        if isinstance(v, (np.ndarray, torch.Tensor)) and v.numel() > 1:
            if k == 'miou_macc_aacc':
                v = compute_miou_macc_aacc_from_statistics(v)
            r.append(f'{k}: {tstr(v)}')
        else:
            r.append(f'{k}: {float(v):.3e}')
    return ', '.join(r)


@torch.no_grad()
def val(model, dl, progress=True,
        metrics={'l1': l1_loss, 'l2': l2_loss},
        avg=True, epoch=None, reduction='batch', draw=False
       ):
    if args.distributed and args.local_rank != 0:
        progress = False  ## 

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

            rgbs, boxes, target, mask = d['rgb'], d['box'], d[args.task], d['mask']
            qs = d['q']  # .to(args.dev, non_blocking=True)
            n, k = qs.shape[:2]
            set_equi_boxes(equi_layers, (boxes, qs, mask))

            target = target[::k]
            # mask = d['mask'][::k] if has_mask else None
            mask = mask[::k]

            p = model(rgbs)
            p0 = p[:n*k:k]

        if draw and step % 500 == 0 and args.local_rank == 0:
            i = 0
            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1); plt.imshow(pt2np(rgbs[i*k,:3],args.rgb_norm).astype(np.float32))
            # if target.dtype is torch.int64:
            if 'semantic' in args.task:
                out_dim = int(p.shape[1]) + 1
                plt.subplot(1,3,2); plt.imshow(pt2np(target[i]),vmin=0,vmax=out_dim,cmap='tab20')
                plt.subplot(1,3,3); plt.imshow(pt2np(p[i].argmax(0)+1),vmin=0,vmax=out_dim,cmap='tab20')
            else:
                # _ = target[i][(mask[i] > 0).repeat(3,1,1)]
                # vmin = _.min().item(); vmax = _.max().item()
                vmin=0; vmax=1
                plt.subplot(1,3,2); plt.imshow(pt2np(target[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmin)
                plt.subplot(1,3,3); plt.imshow(pt2np(p[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmax)
            plt.savefig(args.savedir+f'val_ep={epoch}_step={step}.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()

        # n = len(target)
        # n = mask.sum()
        for k, metric in metrics.items():
            if avg:
                md0[k] += metric(p0, target, mask, reduction='batch').mean(0) * n
            md[k] += metric(p[-n:], target, mask, reduction='batch').mean(0) * n
        md['eq'] += equi_layers[-1].loss.mean() * n
        num += n
        # if progress:
        #     dl.set_postfix({k: f'{md[k]/num:.3e}'})

    if progress and print.__name__ == 'myprint':
        print(dl, fileonly=True)
    if args.distributed:
        if args.local_rank == 0: print('all_reduce...')
        num = torch.tensor(num, device=args.dev)
        torch.distributed.all_reduce(num)
        num = num.item()
    for k in md:
        if args.distributed:
            torch.distributed.all_reduce(md[k])
        md[k] /= num
        if epoch is not None and args.tb:
            logger.add_scalar(f'loss/val_{k}_ep', md[k], epoch)
    if avg:
        for k in md0:
            if args.distributed:
                torch.distributed.all_reduce(md0[k])
            md0[k] /= num
            if epoch is not None and args.tb:
                logger.add_scalar(f'loss/val_{k}_ep', md0[k], epoch)
        if args.local_rank == 0: print(f"loss: p0 {fmt(md0)}\n      p  {fmt(md)}")
        return md0, md
    if args.local_rank == 0: print(f"val loss: p {fmt(md)}")
    return md


def train(model, dl, dl_val, dl_unl=None, num_epochs=1, equi=True, tag='', val0=True, vale=False, metrics={}, loss_fn=l1_loss):
    load = False
    for start_epoch in range(int(np.ceil(num_epochs)),-1,-1):
        _ = args.savedir+f'ep={start_epoch}.pt'
        if os.path.exists(_):
            load = True
        else:
            _ = args.savedir+f'{tag}_ep={start_epoch}.pt'
            if os.path.exists(_):
                load = True
        if load:
            if args.resave:
                s = torch.load(_, map_location='cpu')
                d = s['model'].state_dict()
                if list(d.keys())[0].startswith('module.'):
                    d = {k[len('module.'):]: v for k,v in d.items()}
                s['model'] = d
                torch.save(s, _)
                del s
                print(f'state_dict resaved!: {_}')
                # return False
            else:
                d = torch.load(_, map_location='cpu')['model']  #.state_dict()
                if list(d.keys())[0].startswith('module.'):
                    d = {k[len('module.'):]: v for k,v in d.items()}
            if isinstance(model, nn.parallel.DistributedDataParallel):
                m = model.module
            else:
                m = model
            r = m.load_state_dict(d, strict=False)
            if r.missing_keys:
                print('some keys are missing, try remapping')
                d.update({k: d[k0] for k,k0 in zip(r.missing_keys, r.unexpected_keys)})
                r = m.load_state_dict(d, strict=False)
            if args.local_rank == 0: print(f'resume weights from: {_}, {r}')
            del m, r, d
            break
    # tagb = os.path.basename(tag)

    steps_per_epoch = len(dl)
    total_steps = round(num_epochs*steps_per_epoch)
    num_epochs = int(np.ceil(num_epochs))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(trainable_params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if load:
        assert len(scheduler._last_lr) == 1
        assert len(optimizer.param_groups) == 1
        scheduler.last_epoch = start_epoch * steps_per_epoch
        scheduler._step_count = scheduler.last_epoch + 1
        lr = scheduler._get_closed_form_lr()[0]
        scheduler._last_lr[0] = lr
        for g in optimizer.param_groups:
            g['lr'] = lr
        if args.local_rank == 0: print(f'update lr from {args.lr} to {lr} @ epoch={start_epoch}')

    if args.local_rank == 0:
        print(f"num train params: {len(optimizer.param_groups[0]['params'])}, loss_fn: {loss_fn.__name__}")
        if not args.unlabel:
            args.unl_coef = 0
        print(f'loss = loss_pred + {args.equi_coef} * loss_equi_lab + {args.a_coef} * loss_a + {args.unl_coef} * loss_equi_unl')

    if val0 and start_epoch < num_epochs:
        if args.local_rank == 0: print(tag)
        val(model, dl_val, metrics=metrics, epoch=0, draw=args.local_rank==0)
    t0 = t1 = time()

    scaler = torch.cuda.amp.GradScaler()
    equi_layers = get_equi_layers(model)
    transform = dl.transform

    unl = dl_unl is not None
    if unl:
        def infinite(dl):
            epoch_unl = 0
            while True:
                if args.distributed:
                    dl.sampler.set_epoch(epoch_unl)
                    epoch_unl += 1
                for d in dl:
                    yield d
        it_unl = infinite(dl_unl)

    for epoch in range(start_epoch, num_epochs):
        if args.distributed:
            dl.sampler.set_epoch(epoch)

        model.train()
        set_equi_mode(equi_layers, forward_mode=2)

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
                d = transform(d, color_aug=args.color_aug)

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
                rgbs, targets, boxes, qs, masks = d['rgb'], d[args.task], d['box'], d['q'], d.get('mask', None)

                n_all, k = qs.shape[:2]
                # n, k = batch_size, args.k
                set_equi_boxes(equi_layers, (boxes, qs, masks))

                p = model(rgbs)
                p, pa = p[:n_all*k], p[n_all*k:]

                if unl:
                    # nk1 = n*(k+1)
                    nk = n*k

                    loss_pred = loss_fn(p[:nk], targets[:nk], masks[:nk], reduction='batch').mean()

                    # pa = equi_layers[-1].fa[:n]
                    loss_ave = loss_fn(pa, targets[:nk:k], masks[:nk:k], reduction='batch').mean()

                    if args.sep_batch:
                        # p_lab = p
                        # loss_pred = loss_fn(p_lab, targets, masks, reduction='batch').mean()
                        loss_equi_lab = get_equi_losses(equi_layers).mean()

                        d = du
                        rgbs, targets, boxes, qs, masks = d['rgb'], d[args.task], d['box'], d['q'], d.get('mask', None)
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
                        losses = loss_pred + args.equi_coef * loss_equi

                        loss_pred = loss_pred.mean()
                        # pa = equi_layers[-1].fa
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

                # if args.equi_q:
                #     update_q(losses, qs, transform.sampler.q)
                #     if step % 20 == 0:
                #         q = transform.sampler.q.view(transform.sampler.bins_scale, transform.sampler.bins_ratio,
                #                                      transform.sampler.bins_xy, transform.sampler.bins_xy)
                #         s = ''; names = ['s','r','y','x']
                #         for _ in range(4):
                #             dim = [0,1,2,3]; dim.remove(_)
                #             s += names[_] + ': ' + ','.join('%.4f'%_ for _ in q.mean(dim).numpy()) + '  '
                #         print(s)

            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # n = len(rgbs)
            # n = mask.sum()
            a_loss.update(loss.item(), n)
            if args.task == 'normal':
                p = p.detach().to(torch.float32)
                err = angular_err(p, targets, masks).mean()
                a_1.update(err.item(), n)
            elif args.task == 'segment_semantic':
                # a = accuracy(p, targets, masks)
                a = miou_macc_aacc(p, targets, masks)
                a_1.update(a.clone(), n)

            if equi:
                a_loss_pred.update(loss_pred.item(), n)
                a_loss_ave.update(loss_ave.item(), n)
                if unl:
                    a_loss_equi.update(loss_equi_lab.item(), n)
                    a_loss_equi_unl.update(loss_equi_unl.item(), n_unl)
                else:
                    a_loss_equi.update(loss_equi.item(), n)

            accu_step = epoch*steps_per_epoch+step
            if (step % 100 == 0 or step == steps_per_epoch-1) and args.local_rank == 0:
                t = time()
                if t - t1 > 1200:
                    t1 = t
                    print(f'loss = loss_pred + {args.equi_coef} * loss_equi_lab + {args.a_coef} * loss_a + {args.unl_coef} * loss_equi_unl')
                eta = (t-t0) / (accu_step+1) * (total_steps - accu_step)
                s = f"[ep {epoch:2d} {step:4d}/{steps_per_epoch} lr {lr:.1e} t {(t-t0)/60:.0f}m -{eta/60:.0f}m] loss: {a_loss}"
                if equi:
                    s += f" p: {a_loss_pred}"
                    s += f" a {a_loss_ave}"
                    s += f" eq: {a_loss_equi}"
                    if unl:
                        s += f" u {a_loss_equi_unl}"
                if args.task == 'normal':
                    s += f" err: {a_1}"
                elif args.task == 'segment_semantic':
                    s += f" acc: {a_1}"
                print(s)

                if args.tb:
                    logger.add_scalar('loss/train_loss_step', a_loss.val, accu_step)
                    logger.add_scalar('loss/train_pred_step', a_loss_pred.val, accu_step)
                    logger.add_scalar('loss/train_ave_step', a_loss_ave.val, accu_step)
                    logger.add_scalar('loss/train_equi_step', a_loss_equi.val, accu_step)
                    if unl:
                        logger.add_scalar('loss/train_equi_unl_step', a_loss_equi_unl.val, accu_step)

            if (step % args.img_freq == 0 or step == steps_per_epoch-1) and args.local_rank == 0:
                m = 3
                plt.figure(figsize=(10,9))
                for i in range(3):
                    plt.subplot(3,m,1+i*m); plt.imshow(pt2np(rgbs[i,:3],args.rgb_norm).astype(np.float32))
                    if 'semantic' in args.task:
                        out_dim = int(p.shape[1]) + 1
                        plt.subplot(3,m,2+i*m); plt.imshow(pt2np(targets[i]),vmin=0,vmax=out_dim,cmap='tab20')
                        plt.subplot(3,m,3+i*m); plt.imshow(pt2np(p[i].argmax(0)+1),vmin=0,vmax=out_dim,cmap='tab20')
                    else:
                        vmin=0; vmax=1
                        plt.subplot(3,m,2+i*m); plt.imshow(pt2np(targets[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmax)
                        plt.subplot(3,m,3+i*m); plt.imshow(pt2np(p[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmax)  # vmin=0,vmax=10
                        # _ = targets[i][(masks[i] > 0).repeat(3,1,1)]
                        # vmin = _.min().item(); vmax = _.max().item()
                        # # print(vmin, vmax)
                        # plt.subplot(3,m,2+i*m); plt.imshow(pt2np(targets[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmax)
                        # plt.subplot(3,m,3+i*m); plt.imshow(pt2np(p[i]).astype(np.float32).clip(vmin,vmax),vmin=vmin,vmax=vmax)  # vmin=0,vmax=10
                    if m == 4:
                        plt.subplot(3,m,4+i*m); plt.imshow(pt2np(ws[i]).astype(np.float32),vmin=0,vmax=1)
                plt.savefig(args.savedir+f'ep={epoch}_step={step}.png', dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()

            if accu_step == total_steps-1: break

        if args.tb:
            logger.add_scalar('loss/train_loss_ep', a_loss.avg, epoch+1)
            logger.add_scalar('loss/train_pred_ep', a_loss_pred.avg, epoch+1)
            logger.add_scalar('loss/train_equi_ep', a_loss_equi.avg, epoch+1)
            if unl:
                logger.add_scalar('loss/train_equi_unl_ep', a_loss_equi_unl.avg, epoch+1)

        if (epoch != num_epochs-1 and (epoch+1) % args.val_freq == 0) or (epoch == num_epochs-1 and vale):
            if args.local_rank == 0: print(tag)
            val(model, dl_val, metrics=metrics, epoch=epoch+1, draw=args.local_rank==0)

        if ((epoch+1) % args.save_freq == 0 or epoch == num_epochs-1) and args.local_rank == 0:
            torch.save({'epoch': epoch+1,
                        'model': (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict(),
                        'q': transform.sampler.q},
                       args.savedir+f'ep={epoch+1}.pt')

    return vale



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('arguments for training')
    parser.add_argument('--dir', type=str, default='logs_iccv23')
    parser.add_argument('--tag', type=str,
                        # default='k3_eq0.0001',
                        default='k3_eq0',
                        # default='k1'
                        # default='k0'
                       )
    parser.add_argument('--task', type=str,
                        default='depth_zbuffer'
                        # default='normal'
                        # default='edge_texture'
                        # default='segment_semantic'
                       )
    # parser.add_argument('--split', type=str, default='fast')
    parser.add_argument('--num_bldg', type=int,
                        # default=3,
                        default=100,
                       )
    # parser.add_argument('--percent', type=float, default=33)
    parser.add_argument('--label', type=int,
                        default=None,
                        # default=5000,
                       )
    parser.add_argument('--unlabel', type=int, default=0)
    # parser.add_argument('--unl_batch', type=int, default=1)
    parser.add_argument('--val', type=str,
                        default='tiny',
                        # default='c'
                       )
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--k', type=int,
                        # default=0,
                        # default=1,
                        default=3
                       )
    parser.add_argument('--pos_encode', type=int,
                        default=0,
                        # default=1,
                       )
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--box_type', type=str,
                        default='s0.4-1,r,op0.2'
                        # default='s0.5-1.2,r,op0.1'
                       )
    # parser.add_argument('--var_reduct', type=int, default=0)
    parser.add_argument('--a_coef', type=float, default=0)
    parser.add_argument('--equi_coef', type=float,
                        # default=0.01,
                        default=0.0001,
                        # default=0.
                       )
    parser.add_argument('--unl_coef', type=float,
                        # default=0.0001,
                        default=0.
                       )
    parser.add_argument('--equi_q', type=int, default=0,
                        help='imp samp q: 0-no, 1-learn q IS dist that favors hard crops')
    # parser.add_argument('--q_lr', type=float, default=0.0)
    parser.add_argument('--load_q', type=str, default=None)
    # parser.add_argument('--equi_sg', type=int, default=0, help='stop_grad on avg target')
    parser.add_argument('--equi_loss_mode', type=str, default='forward',
                        help='loss form: forward |f*t - t*fa|, backward |fa - t-1*f*t|')
    parser.add_argument('--equi_layers', type=str,
                        # default='last_act',
                        # default='last_conv2',
                        default='last_conv1',
                        help='last_conv2,last_bn,last_conv1,mid_conv3,up_blocks')
    # parser.add_argument('--lstsq', type=int, default=0)
    parser.add_argument('--net', type=str,
                        default='unet6',
                        # default='resnet',
                        # default='resnet-pt',
                        help='simple,unet,resnet')
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--val0', type=int,
                        default=0
                        # default=1
                       )
    parser.add_argument('--val_freq', type=int,
                        # default=10,
                        # default=1,
                        default=4,
                       )
    parser.add_argument('--save_freq', type=int,
                        default=10,
                        # default=1,
                        # default=2,
                        # default=4,
                       )
    parser.add_argument('--img_freq', type=int, default=5000)
    parser.add_argument('--init', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_fp16', type=int, default=1)
    parser.add_argument('--val_fp16', type=int, default=0)
    parser.add_argument('--sep_batch', type=int, default=0)
    parser.add_argument('--test_only', type=int, default=0)
    parser.add_argument('--color_aug', type=int, default=1)
    parser.add_argument('--color_strength', type=float, default=0.4)
    parser.add_argument('--hue_ratio', type=float, default=0.25)
    parser.add_argument('--rgb_norm', type=int, default=0)
    parser.add_argument('--val_color_aug', type=int, default=0)
    parser.add_argument('--predictor', type=int,
                        default=1
                        # default=0
                       )
    parser.add_argument('--tb', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--resave', type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--local_world_size", type=int, default=None)

    # args = parser.parse_args([])
    args = parser.parse_args()
    # return args

    if "WORLD_SIZE" in os.environ:
        # if args.local_world_size is not None:
        # args.local_rank = int(os.environ["LOCAL_RANK"])

        # These are the parameters used to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
        }
        gpus = args.dev.split(',')
        assert int(env_dict["WORLD_SIZE"]) <= len(gpus)
        args.local_rank = int(env_dict["LOCAL_RANK"])

        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
        args.distributed = True
        dev = int(gpus[args.local_rank])
    else:        
        # if len(gpus) > 1:
        #     print(f'multi-gpu: {gpus}. spawn processes...')
        #     import torch.multiprocessing as mp
        #     mp.spawn(main, nprocs=len(gpus))
        #     exit(0)
        args.distributed = False
        args.local_rank = 0
        dev = int(args.dev.split(',')[0])
    # demo_basic(local_world_size, local_rank)

    args.dev = dev
    torch.cuda.set_device(dev)
    # dev = 'cuda:' + args.dev

    main(args)

    # Tear down the process group
    if args.distributed:
        dist.destroy_process_group()
