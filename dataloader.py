""" ****************************************** 
    Author: In Kyu Lee
    Deep learning dataloaders are stored here.
    Available:
    - ImageDataset: classification
    - SegDataset: Semantic segmentation
    - slice_loader: load slice information for SegDataset
    - CT_loader: load CT images
    - SlicesDataset: Semantic Segmentation (load all images into memory)
    - check_files: check if ct and mask file exist
****************************************** """ 
import os
from medpy.io import load
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2

import scipy.ndimage

def resample(img, hdr, new_spacing=[1,1,1], new_shape=None):
    # new_shape = (64,64,64)
    if new_shape is None:
        spacing = np.array(hdr.spacing, dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
    # new_spacing = spacing / real_resize_factor
    real_resize_factor = np.array(new_shape) / img.shape
    img = scipy.ndimage.interpolation.zoom(img,real_resize_factor, mode='nearest')
    return img, real_resize_factor

def upsample(img, real_resize_factor):
    img = scipy.ndimage.interpolation.zoom(img,1/real_resize_factor, mode='nearest')
    return img


class LungDataset_3D:
    def __init__(self, subjlist, augmentations=None):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.augmentations = augmentations

    def __len__(self):
        return len(self.subj_paths)

    def __getitem__(self,idx):
        img, hdr = load(self.img_paths[idx])
        img, _ = resample(img,hdr,new_shape=(64,64,64))
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = img[None,:]
        return {
                'image': torch.tensor(img.copy())
                }


class LungDataset_2D:
    def __init__(self,subjlist, slices, augmentations=None, img_size=64):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.img_size = img_size
        self.resize_factor = img_size/512
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self,idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, hdr = load(self.img_paths[slc[0]])
            # Reample 512x512xz -> 64x64xz
            # self.img = resample(self.img,hdr,new_shape=(self.img_size,self.img_size,self.img.shape[2]))
            self.img = (self.img-np.min(self.img))/(np.max(self.img)-np.min(self.img))
            self.pat_num = slc[0]
        img = self.img[:,:,slc[1]]
        img = scipy.ndimage.interpolation.zoom(img,self.resize_factor, mode='nearest')
        img = img[None,:]
        return {
                'image': torch.tensor(img.copy())
                }


def slice_loader(subjlist):
    print('Loading Data')
    subj_paths = subjlist.loc[:,'ImgDir'].values
    img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in subj_paths]
    mask_paths = [os.path.join(subj_path,'ZUNU_vida-airtree.img.gz') for subj_path in subj_paths]
    slices = []
    for ii in range(len(mask_paths)):
        label,_ = load(mask_paths[ii])
        img,_ = load(img_paths[ii])
        if img.shape != label.shape:
            print('Dimension does not match: ')
            print(subjlist.loc[ii,'ImgDir'])
        for jj in range(label.shape[2]):
            slices.append((ii,jj))
    return slices

"""
Prepare train & valid dataloaders
"""
def prep_dataloader(c,n_case=0,LOAD_ALL=False):
# n_case: load n number of cases, 0: load all
    df_subjlist = pd.read_csv(os.path.join(c.data_path,c.in_file),sep='\t')
    if n_case==0:
        df_train, df_valid = model_selection.train_test_split(
                df_subjlist,
                test_size=0.2,
                random_state=42,
                stratify=None)
    else:
        df_train, df_valid = model_selection.train_test_split(
             df_subjlist[:n_case],
             test_size=0.2,
             random_state=42,
             stratify=None)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    if LOAD_ALL:
        train_loader = DataLoader(SlicesDataset(CT_loader(df_train,
            mask_name=c.mask)),
            batch_size=c.train_bs, 
            shuffle=True,
            num_workers=0)
        valid_loader = DataLoader(SlicesDataset(CT_loader(df_valid,
            mask_name=c.mask)),
            batch_size=c.valid_bs, 
            shuffle=True,
            num_workers=0)
    else:
        train_slices = slice_loader(df_train)
        valid_slices = slice_loader(df_valid)
        train_ds = SegDataset(df_train,
                              train_slices,
                              mask_name=c.mask,
                              augmentations=get_train_aug())
        valid_ds = SegDataset(df_valid, valid_slices, mask_name=c.mask)
        train_loader = DataLoader(train_ds,
                                  batch_size=c.train_bs,
                                  shuffle=False,
                                  num_workers=0)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=c.valid_bs,
                                  shuffle=False,
                                  num_workers=0)

    return train_loader, valid_loader

