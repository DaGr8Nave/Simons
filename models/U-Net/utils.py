import torch
import numpy as np
import torchvision
from PIL import Image
from dataset import VideoFrameDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
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

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float().cpu()
            preds = torch.squeeze(preds)
        real_image = np.zeros((480, 854, 3))
        for k in range(13):
            print(preds.shape)
            print(preds[k].shape)
            print((preds[k] == 1).shape)
            real_image[preds[k] == 1] = rgb_val[k]
        real_image = torch.tensor(real_image)
        torchvision.utils.save_image(
            real_image, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
