import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models.UNet.unet_parts import *
from models.UNet.unet_model import UNet
from models.NestedUNet.nestedUNet import *
from models.TransUNet.vit_seg_configs import *
from models.TransUNet.vit_seg_modeling import *
from models.TransUNet.vit_seg_modeling_resnet_skip import *
from models.Segmenter.segmenter import *
from models.Segmenter.factory import *
from models.Segmenter.utils import *
from models.Segmenter import config
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset
import os 
from sklearn.model_selection import train_test_split
import numpy as np
from models.DiceLoss import *
from models.topoloss import *
import matplotlib.pyplot as plt

from models.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-3
DEVICE = "cuda"
BATCH_SIZE = 2
NUM_EPOCHS = 20
LOAD_MODEL = False

LAMBDA = 1e-4
loss_per_epoch = []
def train_fn(loader, model, optimizer, loss_fn, scaler, weights=None):
    loop = tqdm(loader)
    current_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = torch.softmax(model(data),dim=1)
            targets = torch.permute(targets,  (0,3,1,2))
            topo = 0
            #for b in range(data.shape[0]):
                #for c in range(predictions.shape[1]):
                    #topo += getTopoLoss(predictions[b,c,:,:], targets[b,c,:,:], topo_size=50) * weights[c] * LAMBDA
            loss = loss_fn(predictions, targets) + topo
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
            A.CenterCrop(480,480),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            #A.Resize(height=240, width=427),
            ToTensorV2(),

        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.CenterCrop(480,480),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            #A.Resize(height=240, width=427),
            ToTensorV2(),
        ],
    )
    CLASSES = 13
    cfg = config.load_config()
    backbone = 'vit_small_patch16_384'
    dataset = 'pascal_context'
    decoder = 'mask_transformer'
    model_cfg = cfg['model'][backbone]
    dataset_cfg = cfg['dataset'][dataset]
    dataset_cfg['im_size']=480
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg
    model = create_segmenter(model_cfg)

    config_vit.patches.grid= (int(480/16), int(480/16))
    config_vit.n_classes=13
    model = VisionTransformer(config_vit, img_size=480, num_classes=13).cuda()
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
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    print(test_dataset.__len__())
    cnts = np.zeros((CLASSES,))
    amts = np.zeros((CLASSES,))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for i, (x,y) in enumerate(train_loader):
        for j in range(CLASSES):
            cnts[j] += y[:,:,:,j].sum()
            if y[:,:,:,j].sum() > 0:
                amts[j]+=1
    print(cnts)
    #for i in range(CLASSES):
        #cnts[i] = cnts[i] * (amts[i]/train_dataset.__len__())

    minimum = np.amin(cnts)
    weights = np.zeros((CLASSES,), dtype=np.float32)
    for i in range(CLASSES):
        weights[i] = minimum/cnts[i]
    print(weights)

    loss_fn = DiceLoss(weight=weights)
    if LOAD_MODEL:
        load_checkpoint(torch.load("../../input/transunetcholecseg8kmodels/TransUNet15ep.pth.tar"), model)
        optimizer.load_state_dict(torch.load("../../input/transunetcholecseg8kmodels/TransUNet15ep.pth.tar")['optimizer'])

    check_accuracy(val_loader, model, device=DEVICE) 
    #save_predictions_as_imgs(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, weights=weights)

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
            #val_loader, model, folder="", device=DEVICE
        #)
    plt.plot(loss_per_epoch)
    plt.show()
if __name__ == "__main__":
    main()
