import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from torch.utils.data import Dataset
import cv2
from PIL import Image


class CustomDataSet(Dataset):

    def __init__(self, main_dir, type='train', resolution=(128,128)):
        self.main_dir = main_dir
        self.root_dir = main_dir
        self.img_dir = os.path.join(self.root_dir,"images")
        self.work_img_dir = os.path.join(self.img_dir,type)
        self.all_imgs = os.listdir(self.work_img_dir)
        self.type = type
        self.compose = transforms.Compose(
            [transforms.Resize(resolution),
             transforms.PILToTensor()]
        )


    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = f'CLEVR_{self.type}_{str(idx).zfill(6)}'
        dir = f'{self.work_img_dir}\\{img_name}.png'
        img = Image.open(dir)
        img = img.convert('RGB')
        tensor_image = self.compose(img)

        #tensor_image = ((tensor_image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
        # tensor_image = self.compose(image)

        return torch.tensor(tensor_image, dtype=torch.float32)


# def main():
#     root_dir = os.path.join(os.path.dirname(os.getcwd()),"CLEVR_v1.0\\CLEVR_v1.0\\")
#     df = CustomDataSet(root_dir)
#     img = df[4]
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()
#
# if __name__ == '__main__':
#     main()