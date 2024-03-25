import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os


class load_data(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.img_dir = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        image_path = self.img_dir[index]
        label = '0' if 'dog' in image_path.split('\\')[-1] else 1
        img_data = np.array(Image.open(image_path))
        img_data_ = torch.from_numpy(img_data)

        return img_data_, label

    def __len__(self):
        return len(self.img_dir)

# torch.set_default_dtype()
ll = load_data('E:\DvsC\data')
img_data,label = ll.__getitem__(0)
print('data:{},\n label:{}'.format(img_data,label))
print('len:{}'.format(ll.__len__()))
# img_data.unsqueeze(0)
img_data_numpy = img_data.numpy()
# plt.imshow(img_data_numpy)


