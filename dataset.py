import os
import torch.utils.data as data
from PIL import Image
from option import args



class MEFdataset(data.Dataset):
    def __init__(self, dir_data, transform):
        super(MEFdataset, self).__init__()
        self.dir_prefix = args.dir_dataset  # 'dataset/'
        self.dir_data = dir_data  # 'train/test'
        self.over = os.listdir(self.dir_prefix + self.dir_data + 'test_over_Y/')
        self.over.sort()
        self.under = os.listdir(self.dir_prefix + self.dir_data + 'test_under_Y/')
        self.under.sort()
        self.transform = transform

    def __len__(self):
        return len(self.over)

    def __getitem__(self, idx):
        over = Image.open(self.dir_prefix + self.dir_data + 'test_over_Y/' + self.over[idx])
        under = Image.open(self.dir_prefix + self.dir_data + 'test_under_Y/' + self.under[idx])

        if self.transform:
            over = self.transform(over)
            under = self.transform(under)

        return over, under

    def save_imgs(self,i):
        save_file = args.save_img
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        out = os.path.join(save_file + self.over[i])
        return out
