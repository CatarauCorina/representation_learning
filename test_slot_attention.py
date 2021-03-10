from slot_attention import SlotAttention
from encoder import AutoEncSlot
import torch
import os
import cv2
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compose = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((128,128)),transforms.PILToTensor()]
)


slot_attention_model = SlotAttention(num_slots=11, dim=128, iters=3).to(device)
enc = AutoEncSlot()
dir = f'{os.getcwd()}/imgs/cats.jpg'
image = torch.tensor(cv2.imread(dir))
image = compose(image)
import matplotlib.pyplot as plt
plt.imshow(image.permute(1,2,0))
plt.show()
image = torch.tensor(image.to(device), dtype=torch.float32).squeeze(0)
out_enc = enc(image.unsqueeze(0))

slot_attention_model(image.unsqueeze(0))
