import os
import torch
import torch.nn as nn
import torch.optim as optim
from slot_attention import SlotAttention
from encoder import AutoEncSlot
from clevr_dataloader import CustomDataSet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

loss = nn.MSELoss()


def l2_loss(predicted, target):
    output = loss(predicted, target)
    return output


def train(model, nr_epochs, optimizer, writer, device):
    for e in range(nr_epochs):
        optimizer.zero_grad()
        train_step(model, optimizer, writer,e, device)


def train_step(enc, optimizer, writer, epoch, device):

    root_dir = os.path.join(os.path.dirname(os.getcwd()), "CLEVR_v1.0\\CLEVR_v1.0\\")
    df = CustomDataSet(root_dir)
    dl = DataLoader(dataset=df, batch_size=10)
    for i_batch, sample_batched in enumerate(dl):
        print(i_batch)
        sample_batched = sample_batched.to(device)
        recon_combined, recons, masks, slots = enc(sample_batched)
        loss_value = l2_loss(recon_combined, sample_batched)/10
        loss_value.backward()
        optimizer.step()
    writer.add_scalar('Loss train', loss_value, epoch)
    torch.save(enc.state_dict(),f'model_epoch_{epoch}.pth')
    return


def main():
    writer = SummaryWriter(f'results/object_discovery_init')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    enc = AutoEncSlot()

    if device == torch.device('cuda'):
        enc.to(device)
    optimizer = optim.Adam(enc.parameters(), lr=0.0004, weight_decay=0.5)
    train(enc, 1000, optimizer, writer, device)


if __name__ == '__main__':
    main()



