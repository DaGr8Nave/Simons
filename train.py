import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models.UNet.unet_parts import *
from models.UNet.unet_model import UNet
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset
import os 
from sklearn.model_selection import train_test_split
import numpy as np
from models.UNet.DiceLoss import DiceLoss
from models.UNet.topoloss import *
import matplotlib.pyplot as plt

from models.UNet.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 5
NUM_EPOCHS = 15
LOAD_MODEL = False

LAMBDA = 3
loss_per_epoch = []
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    current_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = torch.permute(targets,  (0,3,1,2))
            #print(predictions.shape)
            #print(targets.shape)
            loss = loss_fn(predictions, targets)
            #loss += LAMBDA * getTopoLoss(predictions, targets)
            current_loss += loss.item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    loss_per_epoch.append(current_loss)
def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.ChannelShuffle(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height=240, width=427),
            ToTensorV2(),

        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height=240, width=427),
            ToTensorV2(),
        ],
    )

    model = UNet(n_channels=3, n_classes=13).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_val_paths = []
    test_paths = []
    test_ind = [43, 48, 52, 55]
    PATH = '../../input/cholecseg8k'
    for filename in os.listdir(PATH):
        if int(filename[-2:]) not in test_ind:
            for dirs in os.listdir(os.path.join(PATH, filename)):
                train_val_paths.append(os.path.join(os.path.join(PATH, filename), dirs))
        else:
            for dirs in os.listdir(os.path.join(PATH, filename)):
                test_paths.append(os.path.join(os.path.join(PATH, filename), dirs))
    
    np.random.seed(0)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2)
    

    train_dataset = VideoFrameDataset(train_paths, train_transform)
    val_dataset = VideoFrameDataset(val_paths, val_transforms)
    test_dataset = VideoFrameDataset(test_paths, val_transforms)
    cnts = np.zeros((13,))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    for i, (x,y) in enumerate(train_loader):
        for j in range(13):
            cnts[j] += y[:,:,:,j].sum()
    print(cnts)
    for i, (x,y) in enumerate(val_loader):
        for j in range(13):
            cnts[j] += y[:,:,:,j].sum()
    print(cnts)
    for i, (x,y) in enumerate(test_loader):
        for j in range(13):
            cnts[j] += y[:,:,:,j].sum()
    print(cnts)    
    minimum = np.amin(cnts)
    weights = np.zeros((13,), dtype=np.float32)
    for i in range(13):
        weights[i] = minimum/cnts[i]
    print(weights)

    loss_fn = DiceLoss(weight=weights)
    if LOAD_MODEL:
        load_checkpoint(torch.load("../../input/unetforcholecseg8k/87epWeightedDice.pth.tar"), model)
        optimizer.load_state_dict(torch.load("../../input/unetforcholecseg8k/87epWeightedDice.pth.tar")['optimizer'])

    check_accuracy(val_loader, model, device=DEVICE) 
    save_predictions_as_imgs(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="", device=DEVICE
        )
    plt.plot(loss_per_epoch)
    plt.show()
if __name__ == "__main__":
    main()
