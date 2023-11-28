import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
# from PIL import Image, ImageFilter


class UIEBDataset(data.Dataset):
    def __init__(self, raw_path, label_path, trans_path, image_size=256):
        super(UIEBDataset, self).__init__()
        self.raw_path = raw_path
        self.label_path = label_path
        self.trans_path = trans_path
        self.image_size = image_size
        self.data_infos = self.load()

    def load(self):
        data_infos = []
        for data in os.listdir(self.raw_path):
            data_infos.append({
                "raw_path": os.path.join(self.raw_path, data),
                "label_path": os.path.join(self.label_path, data),
                "trans_path": os.path.join(self.trans_path, data), # transmission
                "filename": data,
            })
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        result = self.data_infos[idx]
        raw = Image.open(result['raw_path']).convert('RGB')
        label = Image.open(result['label_path']).convert('RGB')
        transmission = Image.open(result['trans_path']).convert('L')
        img_w = raw.size[0]
        img_h = raw.size[1]
        raw = tfs.Resize([self.image_size, self.image_size])(raw)

        # A = raw.filter(ImageFilter.GaussianBlur(self.image_size))

        transmission = tfs.Resize([self.image_size, self.image_size])(transmission)
        raw = tfs.ToTensor()(raw)
        label = tfs.ToTensor()(label)
        # A = tfs.ToTensor()(A)
        transmission = tfs.ToTensor()(transmission)

        # A = A[:,100:101,100:101]
        raw = tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(raw)
        label = tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(label)
        return raw, label, transmission, result["filename"], str(img_w)+'_'+str(img_h)
