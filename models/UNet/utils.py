import torch
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from dataset import VideoFrameDataset
from torch.utils.data import DataLoader
CLASS_IDS=[0,1,2,3,4,5,6,7,8,9,10,11,12]
CLASSES = ["Black Background", 'Abdominal Wall', "Liver", 'Gastrointestinal Tract', 'Fat', 'Grasper', 'Connective Tissue', 'Blood', 'Cystic Duct', 'L-hook Electrocautery', 'Gallbladder', 'Hepatic Vein', 'Liver Ligament']
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    #print((y_true_f*y_pred_f).shape)
    prod = torch.mul(y_true_f,y_pred_f)
    intersection = torch.sum(prod)
    smooth = 0.0001
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    scores = []
    if y_true.dim() == 3:
        y_true = torch.nn.functional.one_hot(y_true, num_classes=len(CLASS_IDS))
    if y_pred.dim() == 3:
        y_pred = torch.nn.functional.one_hot(y_pred.to(torch.int64), num_classes=len(CLASS_IDS))
    for index in range(numLabels):
        scores.append(dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]))
    return scores
def formatDice(dice):
    for i in range(len(CLASS_IDS)):
        print(f"Dice score for {CLASSES[CLASS_IDS[i]]}: {float(dice[i])}")
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    dice_score = 0
    batches = 0
    dice_score = np.zeros((len(CLASS_IDS)), dtype=np.float32)
    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device)
            #y = y.to(device)
            preds = torch.zeros((x.shape[0],512, 1024), dtype=torch.float)
            for i in range(0, 512, 256):
                for j in range(0, 1024, 256):
                    square = torch.zeros((x.shape[0], :, 256, 256))
                    square[:, :, 0:min(256, 480-i), 0:min(256, 854-j)] = x[:,:,i:min(480,i+256),j:min(854,j+256)]
                    square = square.cuda()
                    out = nn.functional.softmax(model(square),dim=1)
                    out = torch.argmax(preds, dim=1).float()
                    preds[:, i:i+256, j:j+256] = out
            preds = preds[:,:,0:480,0:854]
            batches += 1
            #print(preds[0, :, 395, 205])
            #print(preds.shape)
            #_, ind = torch.max(preds, dim = 1)
            labels = torch.argmax(y, dim=3)
            num_correct += (preds == labels).sum()
            num_pixels += torch.numel(preds)

            if preds.dim() != y.dim():
                preds = torch.nn.functional.one_hot(preds.to(torch.int64), num_classes=len(CLASS_IDS))

            dices = dice_coef_multilabel(y, preds, len(CLASS_IDS))
            #print(dices)
            for i in range(len(CLASS_IDS)):
                dice_score[i] += dices[i]
            #dice_score += (2 * (preds * y).sum()) / (
                #(preds + y).sum() + 1e-8
            #)
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    #print(dice_score)
    dice_score = dice_score/batches
    formatDice(dice_score)
    #print(f"Dice score: {dice_score/batches}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="", device="cuda"
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
            #5, 13, 480, 854
            preds = nn.functional.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1).float().cpu()
            #preds = torch.squeeze(preds)
        #print(preds.shape) #5, 480, 854
        real_image = np.zeros((5, 480, 854, 3), dtype=np.uint8)
        for k in range(13):
            real_image[preds == k] = rgb_val[k]
        for k in range(5):
            img = Image.fromarray(real_image[k])
            img.save(f"{folder}{idx}{k}.png")
            #z.save(f"{folder}{idx}{k}.png")

    model.train()
