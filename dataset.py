import os
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import config as c

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class hidden_dataset(Dataset):
    def __init__(self, root, filename, transforms_=None):
        self.transform = transforms_
        self.root = root
        if filename == 'DIV2K':
            self.picture = os.listdir(self.root)

    def __len__(self):
       return len(self.picture)

    def __getitem__(self, item):
        pictiure_root = os.path.join(self.root, self.picture[item])
        picture = Image.open(pictiure_root)
        picture = to_rgb(picture)
        picture = self.transform(picture)
        return picture

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])
val_transform = T.Compose([
    T.CenterCrop(c.cropsize_val_div2k),
    T.ToTensor()
])
mea_transform = T.Compose([
    T.CenterCrop(c.cropsize_mea_div2k),
    T.ToTensor()
])

train_dataset = hidden_dataset('/storage/student2/xiao/ZKH/datasets/DIV2K_train/DIV2K_train_HR', 'DIV2K', transform)
trainloader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True)
test_dataset = hidden_dataset('/storage/student2/xiao/ZKH/datasets/CoCo', 'DIV2K', val_transform)
testloader = DataLoader(test_dataset, batch_size=c.val_batch_size, shuffle=True, drop_last=True)
measure_dataset = hidden_dataset('/storage/student2/xiao/ZKH/datasets/DIV2K_valid/DIV2K_valid_HR', 'DIV2K', mea_transform)
measureloader = DataLoader(measure_dataset, batch_size=c.mea_batch_size, shuffle=False, drop_last=True)

#/storage/student2/xiao/ZKH/datasets/CoCo
#/storage/student2/xiao/ZKH/datasets/ImageNet









