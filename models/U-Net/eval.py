import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset
import os 
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    dice_coef_multilabel,
    formatDice
)


test_ind = [43, 48, 52, 55]
PATH = '../../../../input/cholecseg8k'
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
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
val_dataset = VideoFrameDataset(val_paths, val_transforms)
test_dataset = VideoFrameDataset(test_paths, val_transforms)
model = UNet(n_channels=3, n_classes=13).to("cuda")
load_checkpoint(torch.load("../../../../input/train-model/Simons/models/U-Net/my_checkpoint.pth.tar"), model)
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
random.seed(0)
for j in range(10):
	print(f"------------------ Results from Prediction {j+1} ------------------")
    i = random.randint(0,test_dataset.__len__())
    #Visualize some results
    x, y = test_dataset.__getitem__(i)
    x = x.to("cuda").unsqueeze(0)
    y = y.to("cuda").unsqueeze(0)
    color_mask = test_dataset.__getcolormask__(i)
    with torch.no_grad():
        preds = nn.functional.softmax(model(x), dim=1)
        #print(preds.shape)
        preds = torch.argmax(preds, dim=1).float()
    #print(f"Dice Score for Prediction {i}: {dice_coef_multilabel(y, preds, 13)}")
    dice = dice_coef_multilabel(y,preds,13)
    formatDice(dice)
    preds = preds.cpu()
    real_image = np.zeros((480, 854, 3), dtype=np.uint8)
    for k in range(13):
        real_image[preds[0] == k] = rgb_val[k]
    real_image = Image.fromarray(real_image)
    original_image = test_dataset.__getimage__(i)
    #original_image, color_mask, real_image
    new_image = Image.new('RGB', (854*3, 480))
    new_image.paste(original_image, (0,0))
    new_image.paste(color_mask, (854,0))
    new_image.paste(real_image, (1708,0))
    new_image.save(f"prediction{i}.png")
