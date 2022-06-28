from torch.utils.data import Dataset
class VideoFrameDataset(Dataset):
    def __init__(self, paths, transforms=None):
        self.image_dir = []
        self.mask_dir = []
        self.transforms = transforms
        for path in paths:
            for filename in os.listdir(path):
                if 'endo_watershed_mask.png' in filename:
                    self.mask_dir.append(os.path.join(path, filename))
                if 'endo.png' in filename:
                    self.image_dir.append(os.path.join(path, filename))
    def __len__(self):
        return len(self.image_dir)
    def __getitem__(self, index):
        #print(self.image_dir[index])
        image = np.array(Image.open(self.image_dir[index]).convert("RGB"))
        mask = np.zeros((13, 480, 854))
        mask_img = np.array(Image.open(self.mask_dir[index]))[:,:,0]
        #print(mask_img.shape)
        for key, value in mapping.items():
            #print(np.where(mask_img==key))
            mask[value][mask_img == key] = 1
        if self.transforms is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
