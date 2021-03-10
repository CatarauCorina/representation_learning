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


def train(model, nr_epochs, optimizer, writer, device, dl, test_dl):
    i =0
    for e in range(nr_epochs):
        optimizer.zero_grad()
        i = train_step(model, optimizer, writer,e, device, dl, test_dl, i)


def eval(model,test_ds, writer, i):
    loss = 0
    iter = 0
    with torch.no_grad():
        for i_batch, img_test in enumerate(test_ds):
            recon_combined, recons, masks, slots = model(img_test)
            loss_value = l2_loss(recon_combined, img_test)
            loss += loss_value
            iter +=10
    writer.add_scalar('Loss test', loss, i)


def train_step(enc, optimizer, writer, epoch, device, dl, test_ds, i):
    print(epoch)
    for i_batch, sample_batched in enumerate(dl):
        sample_batched = sample_batched.to(device)
        recon_combined, recons, masks, slots = enc(sample_batched)
        loss_value = l2_loss(recon_combined, sample_batched)/10
        loss_value.backward()
        optimizer.step()
        writer.add_scalar('Loss iter', loss_value, i)
        if i % 200 == 0 and i != 0:
            eval(enc, test_ds, writer, i)
        i+=1
    writer.add_scalar('Loss train', loss_value, epoch)
    torch.save(enc.state_dict(),f'model_epoch_{epoch}.pth')
    return i


def main():
    root_dir = os.path.join(os.path.dirname(os.getcwd()), "CLEVR_v1.0\\CLEVR_v1.0\\")
    df = CustomDataSet(root_dir)
    train_ds, test_ds = torch.utils.data.random_split(df, [60000, 10000])

    dl_train = DataLoader(dataset=train_ds, batch_size=10)
    dl_test = DataLoader(dataset=test_ds, batch_size=10)
    writer = SummaryWriter(f'results/object_discovery_init100')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    enc = AutoEncSlot()

    if device == torch.device('cuda'):
        enc.to(device)
    optimizer = optim.Adam(enc.parameters(), lr=0.0004, weight_decay=0.5)
    train(enc, 1000, optimizer, writer, device, dl_train,dl_test)


if __name__ == '__main__':
    main()



