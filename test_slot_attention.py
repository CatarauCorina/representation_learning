from slot_attention import SlotAttention
import torch
import os
import cv2
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compose = transforms.Compose(
    [transforms.ToPILImage(),transforms.Resize((128,128)),transforms.PILToTensor()]
)


slot_attention_model = SlotAttention(num_slots=11, dim=128, iters=3).to(device)
dir = f'{os.getcwd()}/imgs/cats.jpg'
image = torch.tensor(cv2.imread(dir))
image = compose(image)
image.to(device)
slot_attention_model(image.unsqueeze(0))
