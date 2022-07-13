import os
import numpy as np
import cv2 as cv

import config

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class LyftDataset(Dataset):
    def __init__(self, sect="A", split="train", transforms=None):
        super().__init__()
        
        if sect.upper() not in ["A", "B", "C", "D", "E", "F"]:
            raise ValueError(f'sect should be in {["A", "B", "C", "D", "E", "F"]}')
        
        if split not in ["train", "val", "full", "test"]:
            raise ValueError(f'split should be in {["train", "val", "full", "test"]}')
        
        self._load_fnames(sect)
        if transforms == -1:
            self.transforms = T.Compose([T.Resize((config.IMG_SIZE, 
                                                   config.IMG_SIZE)),
                                         T.ToTensor()])
        else:
            self.transforms = transforms    

        if split == "train":
            till = int(config.TRAIN_SIZE*len(self.filenames))
            self.filenames = self.filenames[:till]
            
            till = int(config.TRAIN_SIZE*len(self.masknames))
            self.masknames = self.masknames[:till]
        
        elif split == "val":
            frm = int(config.TRAIN_SIZE*len(self.filenames))
            till = int((config.TRAIN_SIZE+config.VAL_SIZE)*len(self.filenames))
            self.filenames = self.filenames[frm:till]
            
            frm = int(config.TRAIN_SIZE*len(self.masknames))
            till = int((config.TRAIN_SIZE+config.VAL_SIZE)*len(self.masknames))
            self.masknames = self.masknames[frm:till]
            
        elif split == "test":
            frm = int((config.TRAIN_SIZE+config.VAL_SIZE)*len(self.filenames))
            self.filenames = self.filenames[frm:]
            
            frm = int((config.TRAIN_SIZE+config.VAL_SIZE)*len(self.masknames))
            self.masknames = self.masknames[frm:] 
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        # print(self.filenames[index])
        img = cv.cvtColor(cv.imread(self.filenames[index]), cv.COLOR_BGR2RGB)
        img = cv.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        img = np.asarray(img).astype('float')
        
        if self.transforms is not None:
            img = self.transforms(img)
            img = torch.as_tensor(img)
        else:
            img = torch.as_tensor(img) / 255.0
        
        img = img.permute(2,0,1)
        
        masks = []
        mask = cv.cvtColor(cv.imread(self.masknames[index]), cv.COLOR_BGR2RGB)
        
        for i in range(13):
            cls_mask = np.where(mask == i, 255, 0)
            cls_mask = cls_mask.astype('float')
            cls_mask = cv.resize(cls_mask, (config.IMG_SIZE, config.IMG_SIZE))

            masks.append(cls_mask[:,:,0] / 255)
            
        masks = torch.as_tensor(masks, dtype=torch.uint8)    
        
        return (img.float(), masks)
    
        
    def _load_fnames(self, sect):
        if sect == "A":
            mask_ext, org_ext = os.listdir(config.DATA_A_DIR)
            self.filenames = os.listdir(os.path.join(config.DATA_A_DIR, 
                                                     org_ext))
            self.masknames = os.listdir(os.path.join(config.DATA_A_DIR, 
                                                     mask_ext))        
            self.filenames = [os.path.join(config.DATA_A_DIR, org_ext, filename) 
                              for filename in self.filenames if filename[-1] != 'r']
            self.masknames = [os.path.join(config.DATA_A_DIR, mask_ext, filename) 
                              for filename in self.masknames if filename[-1] != 'r']
        
        elif sect == "B":    
            mask_ext, org_ext = os.listdir(config.DATA_B_DIR)
            self.filenames = os.listdir(os.path.join(config.DATA_B_DIR, 
                                                     org_ext))
            self.masknames = os.listdir(os.path.join(config.DATA_B_DIR, 
                                                     mask_ext))        
            self.filenames = [os.path.join(config.DATA_B_DIR, org_ext, filename) 
                              for filename in self.filenames if filename[-1] != 'r']
            self.masknames = [os.path.join(config.DATA_B_DIR, mask_ext, filename) 
                              for filename in self.masknames if filename[-1] != 'r']
        
        elif sect == "C":    
            mask_ext, org_ext = os.listdir(config.DATA_C_DIR)
            self.filenames = os.listdir(os.path.join(config.DATA_C_DIR, 
                                                     org_ext))
            self.masknames = os.listdir(os.path.join(config.DATA_C_DIR, 
                                                     mask_ext))
            self.filenames = [os.path.join(config.DATA_C_DIR, org_ext, filename) 
                              for filename in self.filenames if filename[-1] != 'r']
            self.masknames = [os.path.join(config.DATA_C_DIR, mask_ext, filename) 
                              for filename in self.masknames if filename[-1] != 'r']
        
        elif sect == "D":
            mask_ext, org_ext = os.listdir(config.DATA_D_DIR)
            self.filenames = os.listdir(os.path.join(config.DATA_D_DIR, 
                                                     org_ext))
            self.masknames = os.listdir(os.path.join(config.DATA_D_DIR, 
                                                     mask_ext))
            self.filenames = [os.path.join(config.DATA_D_DIR, org_ext, filename) 
                              for filename in self.filenames if filename[-1] != 'r']
            self.masknames = [os.path.join(config.DATA_D_DIR, mask_ext, filename) 
                              for filename in self.masknames if filename[-1] != 'r']        
        
        elif sect == "E":
            mask_ext, org_ext = os.listdir(config.DATA_E_DIR)
            self.filenames = os.listdir(os.path.join(config.DATA_E_DIR, org_ext))
            self.masknames = os.listdir(os.path.join(config.DATA_E_DIR, mask_ext))
            self.filenames = [os.path.join(config.DATA_E_DIR, org_ext, filename) 
                              for filename in self.filenames if filename[-1] != 'r']
            self.masknames = [os.path.join(config.DATA_E_DIR, mask_ext, filename) 
                              for filename in self.masknames if filename[-1] != 'r']
        
        elif sect == "F":
            exts = os.listdir(config.DATA_DIR)
            self.filenames = []
            self.masknames = []
            for ext in exts:
                mask_ext, org_ext = os.listdir(os.path.join(config.DATA_DIR, 
                                                            ext, ext))
                
                temp1 = os.listdir(os.path.join(config.DATA_DIR, ext, 
                                                ext, org_ext))
                temp2 = os.listdir(os.path.join(config.DATA_DIR, ext, 
                                                ext, mask_ext))
                
                temp1 = [os.path.join(config.DATA_DIR, ext, ext, 
                                      org_ext, filename) for filename in temp1 
                         if filename[-1] != 'r']
                temp2 = [os.path.join(config.DATA_DIR, ext, ext, 
                                      mask_ext, filename) for filename in temp2 
                         if filename[-1] != 'r']
                
                self.filenames += temp1
                self.masknames += temp2