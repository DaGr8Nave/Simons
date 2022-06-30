import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset
import os 
from sklearn.model_selection import train_test_split
import numpy as np

from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs
)


test_ind = [43, 48, 52, 55]
PATH = '../../../../input/cholecseg8k'
test_paths = []
for filename in os.listdir(PATH):
    for dirs in os.listdir(os.path.join(PATH, filename)):
    	if int(filename[-2:]) in test_ind:
        	test_paths.append(os.path.join(os.path.join(PATH, filename), dirs))

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

test_dataset = VideoFrameDataset(test_paths, val_transforms)
model = UNet(n_channels=3, n_classes=13).to(DEVICE)
load_checkpoint(torch.load("../../../../input/notebook3855a0b747/Simons/models/U-Net/my_checkpoint.pth.tar"), model)
test_loader = DataLoader(test_dataset, batch_size=5)

check_accuracy(test_loader, model)

for i in range(50):
	#Visualize some results
	x, y = test_dataset.__getitem__(i)
	color_mask = test_dataset.__getcolormask__(i)
    with torch.no_grad():
        preds = nn.functional.softmax(model(x), dim=0)
        print(preds.shape)
        preds = torch.argmax(preds, dim=0).float().cpu()
    real_image = np.zeros((480, 854, 3), dtype=np.uint8)
    for k in range(13):
        real_image[preds == k] = rgb_val[k]
    real_image = Image.fromarray(real_image)
    original_image = test_dataset.__getimage(i)
    #original_image, color_mask, real_image
    new_image = Image.new('RGB', (854*3, 480))
    new_image.paste(original_image, (0,0))
    new_image.paste(color_mask, (854,0))
    new_image.paste(real_image, (1708,0))
    new_image.save(f"prediction{i}.png")
