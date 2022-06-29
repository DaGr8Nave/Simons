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


LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 5
NUM_EPOCHS = 15
LOAD_MODEL = False
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224 
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = torch.squeeze(model(data))
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
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
            ToTensorV2(),
        ],
    )

    model = UNet(n_channels=3, n_classes=13).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_val_paths = []
    test_paths = []
    test_ind = [43, 48, 52, 55]
    PATH = '../../../../input/cholecseg8k'
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("../../../../input/unetforcholecseg8k/my_checkpoint.pth.tar"), model)


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
        #save_predictions_as_imgs(
         #   val_loader, model, folder="saved_images/", device=DEVICE
        #)


if __name__ == "__main__":
    main()
