from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
mapping = {50:0,
          11: 1,
          21: 2,
          13: 3,
          12: 4,
          31: 5,
          23: 6,
          24: 7,
          25: 8,
          32: 9,
          22: 10,
          33: 11,
          5: 12}
CLASS_IDS = [0,1,2,3,4,5,6,7,8,9,10,11,12]
class VideoFrameDataset(Dataset):
    def __init__(self, paths, transforms=None):
        self.image_dir = []
        self.mask_dir = []
        self.color_mask_dir = []
        self.transforms = transforms
        for path in paths:
            #start_num = int(path[-5:])
            start_num = 0
            for k in range(600):
                curr_num = start_num + k
                curr_num = str(curr_num)
                while len(curr_num) < 6:
                    curr_num = '0'+curr_num
                self.image_dir.append(os.path.join(path, f'{curr_num}.png'))
                self.mask_dir.append(os.path.join(path, f'{curr_num}.png'))
                self.color_mask_dir.append(os.path.join(path, f'{curr_num}.png'))
    def __len__(self):
        return len(self.image_dir)
    def __getitem__(self, index):
        #print(self.image_dir[index])
        #print(self.image_dir[index])
        #print(self.mask_dir[index])
        image = np.array(Image.open(self.image_dir[index]).convert("RGB"))
        mask = np.zeros((480, 854), dtype=np.float32)
        mask_img = np.array(Image.open(self.mask_dir[index]))[:,:,0]
        #print(mask_img.shape)
        for key, value in mapping.items():
            #print(np.where(mask_img==key))
            if value in CLASS_IDS:
                mask[mask_img == key] = CLASS_IDS.index(value)
            else:
                mask[mask_img == key] = 0
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        mask = mask.long()
        mask = torch.nn.functional.one_hot(mask, len(CLASS_IDS))
        return image, mask
    def __getimage__(self, index):
        #no normalization
        image = Image.open(self.image_dir[index])
        return image
    def __getcolormask__(self, index):
        color_mask = Image.open(self.color_mask_dir[index])
        return color_mask
