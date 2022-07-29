#import torch
import argparse
import cv2
from dataset import VideoFrameDataset
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader
from models.UNet.unet_parts import *
from models.UNet.unet_model import UNet
parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--video") 
parser.add_argument("--output")
args = parser.parse_args()
MODEL_PATH = "../" + args.model
VIDEO_PATH = "../" + args.video
OUTPUT_PATH = args.output
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
#make predictions
dataset = VideoFrameDataset([VIDEO_PATH], transforms=val_transforms)
result = cv2.VideoWriter(OUTPUT_PATH, 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         25, (480*2, 480))
loader = DataLoader(dataset, 5, shuffle=False)
model = UNet(n_channels=3, n_classes=13).to("cuda")
model.load_state_dict(torch.load(MODEL_PATH)["state_dict"])
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

for i, (x,y) in enumerate(loader):
    x = x.to("cuda")
    y = y.to("cuda")
    with torch.no_grad():
        preds = nn.functional.softmax(model(x), dim=1)
        #print(preds.shape)
        preds = torch.argmax(preds, dim=1).float()    
    real_image = np.zeros((x.shape[0], 480, 480, 3), dtype=np.uint8)
    preds = preds.cpu()
    for k in range(13):
        real_image[preds == k] = rgb_val[k]
    for k in range(5):
        predicted = Image.fromarray(real_image[k])
        original_image = np.array(dataset.__getimage__(5*i+k))
        #color_mask = np.array(dataset.__getcolormask__(5*i+k))
        transformed = center_crop(image = original_image)
        original_image = Image.fromarray(transformed['image'])
        #color_mask = Image.fromarray(transformed['mask'])        
        new_image = Image.new('RGB', (480*2, 480))
        new_image.paste(original_image, (0,0))
        #new_image.paste(color_mask, (480,0))
        new_image.paste(predicted, (480,0))
        open_cv_image = np.array(new_image)
        open_cv_image = open_cv_image[:, : , ::-1].copy()
        result.write(open_cv_image)
result.release()
