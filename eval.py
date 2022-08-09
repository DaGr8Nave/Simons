import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random
import torch.nn as nn
import torch.optim as optim
from models.UNet.unet_model import UNet
from models.NestedUNet.nestedUNet import *
from models.TransUNet.vit_seg_configs import *
from models.TransUNet.vit_seg_modeling import *
from models.TransUNet.vit_seg_modeling_resnet_skip import *
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset
import os 
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

from models.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    dice_coef_multilabel,
    formatDice
)


test_ind = [43, 48, 52, 55]
PATH = '../../input/cholecseg8k'
test_paths = []
train_val_paths = []
for filename in os.listdir(PATH):
    for dirs in os.listdir(os.path.join(PATH, filename)):
        if int(filename[-2:]) in test_ind:
            test_paths.append(os.path.join(os.path.join(PATH, filename), dirs))
        else:
            train_val_paths.append(os.path.join(os.path.join(PATH,filename),dirs))
np.random.seed(0)
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2)
val_transforms = A.Compose(
    [
        #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.CenterCrop(480,480),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
center_crop = A.Compose(
    [
        A.CenterCrop(480,480),
    ]
    
)
val_dataset = VideoFrameDataset(val_paths, val_transforms)
test_dataset = VideoFrameDataset(test_paths, val_transforms)
CLASSES=13
CLASS_IDS=[0,1,2,3,4,5,6,7,8,9,10,11,12]
config_vit = CONFIGS['R50-ViT-B_16']
config_vit.patches.grid= (int(480/16), int(480/16))
config_vit.n_classes=CLASSES
model = VisionTransformer(config_vit, img_size=480, num_classes=CLASSES).cuda()
load_checkpoint(torch.load("../../input/transunetcholecseg8kmodels/TransUNet35ep.pth.tar"), model)
test_loader = DataLoader(test_dataset, batch_size=5)
val_loader = DataLoader(val_dataset, batch_size=5)
print("------------------ Test Set Results ------------------")
check_accuracy(test_loader, model)
print("------------------ Validation Set Results ------------------")
check_accuracy(val_loader, model)
rgb_val = np.zeros((13, 3))
rgb_val[0] = np.array([127,127,127])
rgb_val[1] = np.array([210,140,140])
rgb_val[2] = np.array([255,114,114])
rgb_val[3] = np.array([231,70,156])
rgb_val[4] = np.array([186,183,75])
rgb_val[5] = np.array([170,255,0])
rgb_val[6] = np.array([255,85,0])
rgb_val[7] = np.array([255,0,0])
rgb_val[8] = np.array([255,255,0])
rgb_val[9] = np.array([169,255,184])
rgb_val[10] = np.array([255,160,165])
rgb_val[11] = np.array([0,50,128])
rgb_val[12] = np.array([111,74,0])
random.seed(4)
for j in range(50):
    print(f"------------------ Dice from Prediction {j+1} ------------------")
    i = random.randint(0, test_dataset.__len__())   
    #Visualize some results
    x, y = test_dataset.__getitem__(i)
    x = x.to("cuda").unsqueeze(0)
    y = y.to("cuda").unsqueeze(0)
    color_mask = np.array(test_dataset.__getcolormask__(i),dtype=np.uint8)
    with torch.no_grad():
        preds = nn.functional.softmax(model(x), dim=1)
        #print(preds.shape)
        preds = torch.argmax(preds, dim=1).float()
    #print(f"Dice Score for Prediction {i}: {dice_coef_multilabel(y, preds, 13)}")
    dice = np.array(dice_coef_multilabel(y,preds,CLASSES))
    print(dice.shape)
    formatDice(np.divide(dice[:,0],dice[:,1]))
    preds = preds.cpu()
    real_image = np.zeros((480, 480, 3), dtype=np.uint8)
    for k in range(CLASSES):
        real_image[preds[0] == k] = rgb_val[CLASS_IDS[k]]
    real_image = Image.fromarray(real_image)
    original_image = np.array(test_dataset.__getimage__(i), dtype=np.uint8)
    transformed = center_crop(image = original_image, mask = color_mask)
    original_image = Image.fromarray(transformed['image'])
    color_mask = Image.fromarray(transformed['mask'])
    
    #original_image, color_mask, real_image
    new_image = Image.new('RGB', (480*3, 480))
    new_image.paste(original_image, (0,0))
    new_image.paste(color_mask, (480,0))
    new_image.paste(real_image, (960,0))
    new_image.save(f"prediction{i}.png")
