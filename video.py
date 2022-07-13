#import torch
import argparse
import cv2
from dataset import VideoFrameDataset
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--video")
parser.add_argument("--output")
args = parser.parse_args()
MODEL_PATH = args.model
VIDEO_PATH = args.video
OUTPUT_PATH = args.output

#make predictions
dataset = VideoFrameDataset([VIDEO_PATH])
result = cv2.VideoWriter(OUTPUT_PATH, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (854*3, 480))
loader = DataLoader(dataset, 5, shuffle=False)
model = torch.load(MODEL_PATH)["state_dict"]
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
    x.to("cuda")
    y.to("cuda")
    with torch.no_grad():
        preds = nn.functional.softmax(model(x), dim=1)
        #print(preds.shape)
        preds = torch.argmax(preds, dim=1).float()    
    real_image = np.zeros((x.shape[0], 480, 854, 3), dtype=np.uint8)
    for k in range(13):
        real_image[preds == k] = rgb_val[k]
    for k in range(5):
        predicted = Image.fromarray(real_image[k])
        original_image = test_dataset.__getimage__(5*i+k)
        color_mask = test_dataset.__getcolormask__(5*i+k)
        new_image = Image.new('RGB', (854*3, 480))
        new_image.paste(original_image, (0,0))
        new_image.paste(color_mask, (854,0))
        new_image.paste(real_image, (1708,0))
        open_cv_image = np.array(new_image)
        open_cv_image = open_cv_image[:, : , ::-1].copy()
        result.write(open_cv_image)
result.release()
