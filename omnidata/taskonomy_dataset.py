from   collections import namedtuple, Counter, defaultdict
from   dataclasses import dataclass, field
import os
import pickle
from   PIL import Image, ImageFile
import re
import torch
import torch.utils.data as data
from   typing import Optional, List, Callable, Union, Dict, Any, Tuple
import warnings
from tqdm import tqdm
from io import BytesIO

from .masks import make_mask, make_mask_from_data, DEFAULT_MASK_EXTRA_RADIUS
from .splits import taskonomy_flat_split_to_buildings
from .transforms import default_loader, get_transform
from .task_configs import task_parameters, SINGLE_IMAGE_TASKS #, *

ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this


LabelFile = namedtuple('LabelFile', ['point', 'view', 'domain'])
View = namedtuple('View', ['building', 'point', 'view'])

                   
class TaskonomyDataset(data.Dataset):
    '''
        Loads data for the Taskonomy dataset.
        This expects that the data is structured
        
            /path/to/data/
                rgb/
                    modelk/
                        point_i_view_j.png
                        ...                        
                depth_euclidean/
                ... (other tasks)
                
        If one would like to use pretrained representations, then they can be added into the directory as:
            /path/to/data/
                rgb_encoding/
                    modelk/
                        point_i_view_j.npy
                ...
        
        Basically, any other folder name will work as long as it is named the same way.
    '''
    @dataclass
    class Options():
        '''
            data_path: Path to data. Either one for all, or a dictionary mapping tasks to their data paths.
            tasks: Which tasks to load. Any subfolder will work as long as data is named accordingly
            buildings: Which models to include. See `splits.taskonomy` (can also be a string, e.g. 'fullplus-val')
            transform: one transform per task.
            
            Note: This assumes that all images are present in all (used) subfolders
        '''
        data_path: Union[str, Dict[str, str]] = '/scratch/group/taskonomy'
        tasks: List[str] = field(default_factory=lambda: ['rgb'])
        buildings: Union[List[str], str] = 'tiny'
        transform: Optional[Union[List[Callable], str]] = "DEFAULT"  # List[Transform], None, "DEFAULT"
        load_to_mem: bool = False
        zip_file_name: bool = False
        return_mask: bool = False
        mask_extra_radius: int = DEFAULT_MASK_EXTRA_RADIUS
        image_size: Optional[int]=None
        target_size: Optional[int]=None
        force_refresh_tmp: bool = True
        bin_root: str = '/tmp/yuanyi/'


    def __init__(self, options: Options, verbose=False):
        self.return_tuple = True
        if isinstance(options.tasks, str):
            options.tasks = [options.tasks]
            options.transform = [options.transform]            
            self.return_tuple = False
        
        self.buildings = taskonomy_flat_split_to_buildings[options.buildings] if isinstance(options.buildings, str) else options.buildings
        self.cached_data = {}
        self.data_path = options.data_path
        self.image_size = options.image_size
        self.target_size = options.target_size
        if self.target_size is None:
            self.target_size = self.image_size
        self.load_to_mem = options.load_to_mem
        self.mask_extra_radius = options.mask_extra_radius
        self.return_mask = options.return_mask
        self.tasks = options.tasks
        self.zip_file_name = options.zip_file_name
        self.force_refresh_tmp = options.force_refresh_tmp
        self.bin_root = options.bin_root

        # Load saved image locations if they exist, otherwise create and save them
        tmp_path = './tmp/taskonomy_{}_{}.pkl'.format(
            '-'.join(options.tasks), 
            '-'.join([b[:3] for b in options.buildings]) if isinstance(options.buildings, list) else options.buildings
        )
        tmp_exists = os.path.exists(tmp_path)
        if tmp_exists and not self.force_refresh_tmp:
            with open(tmp_path, 'rb') as f:
                self.urls = pickle.load(f)
            self.size = len(self.urls[self.tasks[0]])
            print(f'Loaded TaskonomyDataset with {self.size} images from tmp.')
        else:
            self.urls = {task: make_dataset(
                            os.path.join(self.data_path, task, 'taskonomy'), self.buildings, self.bin_root, verbose)
                        for task in options.tasks}
            # self.urls = {task: make_dataset(os.path.join(self.data_path, task), self.buildings)
            self.urls, self.size  = self._remove_unmatched_images()

            # Save extracted URLs
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, 'wb') as f:
                pickle.dump(self.urls, f)

        self.pid = None
        self.bin_files = {}

        self.transform = options.transform
        if isinstance(self.transform, str):
            if self.transform == 'DEFAULT':
                self.transform = [get_transform(task, self.image_size if 'rgb' in task else self.target_size) for task in self.tasks]
            else:
                raise ValueError('TaskonomyDataset option transform must be a List[Callable], None, or "DEFAULT"')

        # Perhaps load some things into main memory
        # Note (March 2020: This is likely broken due to the transforms not necessarily being a list)
        if self.load_to_mem: 
            print('Writing activations to memory')
            for t, task in zip(self.transform, self.tasks):
                self.cached_data[task] = [None] * len(self)
                # for i, url in enumerate(tqdm(self.urls[task],desc=task)):
                #     self.cached_data[task][i] = t(default_loader(url))
                # self.cached_data[task] = torch.stack(self.cached_data[task])
#             self.cached_data = torch.stack(self.cached_data)
            print('Finished writing some activations to memory')


    def ensure_file_open(self, path):
        pid = os.getpid()
        if self.pid != pid or path not in self.bin_files:
            self.pid = pid
            self.bin_files[path] = open(os.path.join(self.bin_root,path),'rb')
        return self.bin_files[path]

    def default_loader(self, task, path):
        if isinstance(path, tuple):
            path, offset, size = task+'_small/'+path[0].split('/')[0], path[1], path[2]
            f = self.ensure_file_open(path+'.binary')
            f.seek(offset)
            b = BytesIO(f.read(size))
            img = Image.open(b)
            img = img.convert(img.mode)
            img.load()
            b.close()
            return img
        return default_loader(path)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        mask, mask_needed = None, self.return_mask
        fpaths = [self.urls[task][index] for task in self.tasks]

        # if self.load_to_mem:
        #     result = tuple([self.cached_data[task][index] for task in self.tasks])
        # else:
        #     result = [default_loader(path) for path in fpaths]
        if self.load_to_mem and self.cached_data[self.tasks[0]][index] is not None:
            result = tuple([self.cached_data[task][index] for task in self.tasks])
        else:
            result = [self.default_loader(task, path) for task, path in zip(self.tasks,fpaths)]
            if self.load_to_mem:
                for task, tensor in zip(self.tasks, result):
                    self.cached_data[task][index] = tensor
        # yyz: unindent
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                # result = [transform(tensor) for transform, tensor in zip(self.transform, result)]
                result_post = []
                for i, (transform, tensor) in enumerate(zip(self.transform, result)):
                    # try:
                    if transform is not None:
                        tensor = transform(tensor)
                    result_post.append(tensor)
                    if mask_needed and 'mask_val' in task_parameters[self.tasks[i]]:
                        mask = make_mask(tensor, self.tasks[i])
                        mask_needed = False
                    # except Exception as e:
                    #     print('exception in task:', self.tasks[i], transform, tensor)
                    #     raise e
                result = result_post
            else:
                result = self.transform(result)
                if mask_needed:
                    for i in range(len(self.tasks)):
                        if 'mask_val' in task_parameters[self.tasks[i]]:
                            mask = make_mask(result[i], self.tasks[i])
                            mask_needed = False
                            break

        # handle 2 channel outputs
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            # yyz: what a shitty code...
            # base_task = [t for t in SINGLE_IMAGE_TASKS if t == task]
            # if len(base_task) == 0:
            #     continue
            # else:
            #     base_task = base_task[0]
            # num_channels = task_parameters[base_task]['out_channels']
            if task not in SINGLE_IMAGE_TASKS:
                continue
            num_channels = task_parameters[task]['out_channels']
            if 'decoding' in task and result[i].shape[0] != num_channels:
                assert torch.sum(result[i][num_channels:,:,:]) < 1e-5, 'unused channels should be 0.'
                result[i] = result[i][:num_channels,:,:]

        result_dict = {task: value for task, value in zip(self.tasks, result)}

        if self.return_mask:
            if mask_needed:
                raise NotImplementedError('Mask not created!')
            result_dict['mask'] = mask
    
        if self.zip_file_name:
            for fpath, task in zip(fpaths, self.tasks):
                result_dict[task + '_fpath'] = fpath 

        if self.return_tuple:
            return result_dict
        else:
            return result[0]

    # def make_mask(self, data: Dict[str, Any]):
    #     ''' data: Map of tasks -> Tensors '''
    #     tasks = sorted(data.keys())
    #     return make_mask_from_data(
    #                         [data[t] for t in tasks],
    #                         tasks,
    #                         self.mask_extra_radius)
    
    def task_config(self, task):
        return task_parameters[task]


    def _remove_unmatched_images(self) -> (Dict[str, List[str]], int):
        '''
            Filters out point/view/building triplets that are not present for all tasks
            
            Returns:
                filtered_urls: Filtered Dict
                max_length: max([len(urls) for _, urls in filtered_urls.items()])
        '''
        n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
        max_images = max(n_images_task)[0]
        if max(n_images_task)[0] == min(n_images_task)[0]:
            return self.urls, max_images
        else:
            print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))
            # Get views for each task
            
            def _parse_fpath_for_view( path ):
                if isinstance(path, tuple):
                    path = path[0]
                building = os.path.basename(os.path.dirname(path))
                file_name = os.path.basename(path) 
                lf = parse_filename( file_name )
                return View(view=lf.view, point=lf.point, building=building)

            self.task_to_view = {}
            for task, paths in self.urls.items():
                self.task_to_view[task] = [_parse_fpath_for_view( path ) for path in paths]
            
            # Compute intersection
            intersection = None
            for task, uuids in self.task_to_view.items():
                if intersection is None:
                    intersection = set(uuids)
                else:
                    intersection = intersection.intersection(uuids)

#             print(sorted(list(intersection)))
            # Keep intersection
            print('Keeping intersection: ({} images/task)...'.format(len(intersection)))
            new_urls = {}
            for task, paths in self.urls.items():
                new_urls[task] = [path for path in paths if _parse_fpath_for_view( path ) in intersection]
#                 for path in paths:
#                     building = os.path.basename(os.path.dirname(path))
#                     file_name = os.path.basename(path) 
#                     lf = parse_filename( file_name )
#                     view = str(View(view=lf.view, point=lf.point, building=building))
#                     print(view)
#                     if view in intersection:
#                         paths_for_task.append(path)
#                 new_urls[task] = paths_for_task
            return new_urls, len(intersection)
        raise NotImplementedError('Reached the end of this function. You should not be seeing this!')

    
    def _validate_images_per_building(self):
            # Validate number of images
            print("Building TaskonomyDataset:")
            n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
            print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
            if max(n_images_task)[0] != min(n_images_task)[0]:
                print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                    max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))

                # count number of frames per building per task
                all_building_counts = defaultdict(dict)
                for task, obs in self.urls.items():
                    c = Counter([url.split("/")[-2] for url in obs])
                    for building in c:
                        all_building_counts[building][task] = c[building]

                # find where the number of distinct counts is more than 1
                print('Removing data from the following buildings')
                buildings_to_remove = []
                for b, count in all_building_counts.items():
                    if len(set(list(count.values()))) > 1:
                        print(f"\t{b}:", count)
                        buildings_to_remove.append(b)
                    if len(count) != len(self.tasks):
                        print(f"\t{b}: missing in tasks", set(self.tasks) - set(count.keys()))
                        buildings_to_remove.append(b)
                # [(len(obs), task) for task, obs in self.urls.items()]

                # redo the loading with fewer buildings
                buildings_redo = [b for b in self.buildings if b not in buildings_to_remove]
                self.urls = {task: make_dataset(os.path.join(self.data_path, task, 'taskonomy'), buildings_redo)
                            for task in self.tasks}
                n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
                print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
            assert max(n_images_task)[0] == min(n_images_task)[0], \
                    "Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                    max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task]))
            return n_images_task


def make_dataset(dir, folders=None, bin_root=None, verbose=False):
    # folders are building names. If None, get all the images (from both building folders and dir)
    # prefer loading the binary formats for faster reads
    images = []

    if bin_root:
        task = dir.split('/')[-2]
        b = None
        bin_folder = os.path.join(bin_root, task+'_small/')
        if os.path.exists(bin_folder):
            if verbose: print('Binary format exists:', bin_folder)
            for file in sorted(os.listdir(bin_folder)):
                subfolder = file.split('.')[0]
                if file.endswith('.label') and (folders is None or subfolder in folders):
                    with open(f'{bin_folder}{file}', 'r') as f:
                        lines = [_.split('\t') for _ in f.read().splitlines()]
                        lines = sorted(lines)
                        lines = [(subfolder+'/'+_[0],int(_[1]),int(_[2])) for _ in lines]
                        # b_images = [subfolder+'/'+_[0] for _ in lines]
                        # b_meta = [(int(_[1]),int(_[2])) for _ in lines]
                    images += lines
            return images

    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for subfolder in sorted(os.listdir(dir)):
        subfolder_path = os.path.join(dir, subfolder)
        if os.path.isdir(subfolder_path) and (folders is None or subfolder in folders):
            for fname in sorted(os.listdir(subfolder_path)):
                path = os.path.join(subfolder_path, fname)
                images.append(path)

        # If folders/buildings are not specified, use images in dir
        if folders is None and os.path.isfile(subfolder_path):
            images.append(subfolder_path)

    return images


def parse_filename( filename ):
    p = re.match('.*point_(?P<point>\d+)_view_(?P<view>\d+)_domain_(?P<domain>\w+)', filename)
    if p is None:
        raise ValueError( 'Filename "{}" not matched. Must be of form point_XX_view_YY_domain_ZZ.**.'.format(filename) )

    lf = {'point': p.group('point'), 'view': p.group('view'), 'domain': p.group('domain') }
    return LabelFile(**lf)


class TaskonomyDataLoader:

    @dataclass
    class Options(TaskonomyDataset.Options):
        phase: str = 'val'
        batch_size: int = 6
        shuffle: bool = True
        num_workers: int = 8
        pin_memory: bool = True
        collate_fn = None

    def make(options: Options = None, dataset = None, sampler=None):
        is_train = (options.phase == 'train')
        if dataset is None:
            dataset = TaskonomyDataset(options)
        if sampler: assert not options.shuffle
        return data.DataLoader(
            dataset=dataset,
            batch_size=options.batch_size,
            shuffle=options.shuffle,
            num_workers=options.num_workers,
            pin_memory=options.pin_memory,
            drop_last=is_train,
            collate_fn=options.collate_fn,
            persistent_workers=True,
            sampler=sampler)
