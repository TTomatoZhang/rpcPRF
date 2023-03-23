"""
This script defines the dataloader for a datasets of multi-view satellite images
"""
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from datasets.utils_pushbroom import *
import cv2
import tqdm

import rasterio
import glob


#from datasets.rpcm import *

class MVSDataset_IARPA(Dataset):
    def __init__(self, conf, split="train"):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
        """
        self.conf = conf
        self.split = split

        self.num_view = conf['general.neighbor_view_num']

        if split == 'train' or split == 'val':
            self.root_dir = os.path.join(conf['dataset.root_dir'], 'Train_ex', conf['dataset.dname'])
        else:
            self.root_dir = os.path.join(conf['dataset.root_dir'], 'Test_ex', conf['dataset.dname'])
        assert os.path.exists(self.root_dir), f"root_dir {self.root_dir} does not exist"

        self.imgH = conf['dataset.imgH']
        self.imgW = conf['dataset.imgW']


        self.impath = os.path.join(self.root_dir, 'img')
        self.rpcpath = os.path.join(self.root_dir, 'rpc_aug')
        #self.rpcpath = os.path.join(self.root_dir, 'augrpc')
        self.heipath = os.path.join(self.root_dir, 'alt')
        sub_ls = os.listdir(self.impath)
        self.scenes = [subpath.split('/')[-1] for subpath in sub_ls]
        #self.numfile = 'group{}.txt'.format(self.num_view)
        for scene in self.scenes:
            if not os.path.exists(os.path.join(self.impath, scene, 'group{}.txt'.format(self.num_view))):
                self.scenes.remove(scene)

        self.tiftriples, self.rpctriples, self.heitriples = [], [], []

        self.numpairs = 0

            #
            # tifglob = glob.glob(os.path.join(self.impath, scene, '/*.tif'))
            # rpcglob = glob.glob(os.path.join(self.rpcpath, scene, '/*.txt'))
            # heiglob = glob.glob(os.path.join(self.heipath, scene, '/*.tif'))

            # self.scenelist = os.listdir(self.rpcpath)
        self.parsepairs()
        print('finish loading all scene!')

    def parsepairs(self):
        for scene in self.scenes:
            path = os.path.join(self.impath, scene, 'group{}.txt'.format(self.num_view))
            #path = os.path.join(self.impath, scene, 'pair.txt'.format(self.num_view))
            with open(path) as f:
                alltext = f.read().splitlines()
            f.close()
            numpair = int(alltext[0])
            self.numpairs += numpair
            for i in range(numpair):
                # pair_ls = [int(at) for at in alltext[1:].split(" ")]
                pairname = alltext[i + 1].split(" ")
                while "" in pairname:
                    pairname.remove("")
                tiftriple = [os.path.join(self.impath, scene, '{}.tif'.format(name)) for name in pairname]
                rpctriple = [os.path.join(self.rpcpath, scene, '{}_aug.rpc'.format(name)) for name in pairname]
                #rpctriple = [os.path.join(self.rpcpath, scene, '{}.txt'.format(name)) for name in pairname]
                heitriple = [os.path.join(self.heipath, scene, '{}.pfm'.format(name)) for name in pairname]

                self.tiftriples.append(tiftriple)
                self.rpctriples.append(rpctriple)
                self.heitriples.append(heitriple)
            print('finish loading triple paths of ' + scene)



    def __len__(self):
        # compute length of dataset
        return self.numpairs

    def __getitem__(self, idx):
        tiftriple, rpctriple, heitriple = self.tiftriples[idx], self.rpctriples[idx], self.heitriples[idx]
        rgb_src = load_tensor_from_rgb_geotiff(tiftriple[0]).unsqueeze(0).permute(0, 3, 1, 2)
        rgbs_tgt = [load_tensor_from_rgb_geotiff(rgbpath).unsqueeze(0).permute(0, 3, 1, 2) for rgbpath in tiftriple[1:]]

        rpc_src = load_aug_rpc_tensor_from_txt(rpctriple[0])
        rpcs_tgt = [load_aug_rpc_tensor_from_txt(rpcpath).unsqueeze(0) for rpcpath in rpctriple[1:]]

        # max_src, min_src = GetH_MAX_MIN(rpc_src)
        # alt_src = load_gray_tensor_from_geotiff(heitriple[0]).unsqueeze(0)

        #alt_src, max_src, min_src = load_gray_tensor_maxmin_from_geotiff(heitriple[0])
        alt_src = torch.from_numpy(load_pfm(heitriple[0]).copy())
        alt_src = alt_src.unsqueeze(0)
        max_src, min_src = torch.max(alt_src), torch.min(alt_src)

        # alts_tgt = [load_gray_tensor_from_geotiff(heipath) for heipath in heitriple[1:]]
        alts_tgt = [torch.from_numpy(load_pfm(heipath).copy()).unsqueeze(0) for heipath in heitriple[1:]]

        if self.split == 'test':
            def getname(path):
                namesplits = path.split('/')
                base = namesplits[-1].split('.')[0]
                return namesplits[-2] + '_' + base
            names = [getname(rgbpath) for rgbpath in tiftriple]
            sample = {
                "image_src": rgb_src,  # torch Tensor: []
                "images_tgt": torch.cat((rgbs_tgt), dim=0),  # torch Tensor
                "rpc_src": rpc_src,  # torch Tensor
                "rpcs_tgt": torch.cat((rpcs_tgt), dim=0),  # torch Tensor
                "alt_min_src": min_src,  # float
                "alt_max_src": max_src,  # float
                "alt_src": alt_src,  #
                # [nNeighbor, 3, H, W], np.array
                "alts_tgt": torch.cat((alts_tgt), dim=0),  # [nNeighbor, H, W], torch Tensor
                "names": names
            }
        else:
            sample = {
                "image_src": rgb_src,  # torch Tensor: []
                "images_tgt": torch.cat((rgbs_tgt), dim=0),  # torch Tensor
                "rpc_src": rpc_src,  # torch Tensor
                "rpcs_tgt": torch.cat((rpcs_tgt), dim=0),  # torch Tensor
                "alt_min_src": min_src,  # float
                "alt_max_src": max_src,  # float
                "alt_src": alt_src,  #
                  # [nNeighbor, 3, H, W], np.array
                "alts_tgt": torch.cat((alts_tgt), dim=0)  # [nNeighbor, H, W], torch Tensor
            }

        return sample



