import os
import torch
import torch.nn as nn
import torch.optim as optim
from slot_attention import SlotAttention
from encoder import AutoEncSlot
from clevr_dataloader import CustomDataSet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib.pyplot import Line2D

import matplotlib.pyplot as plt

loss = nn.MSELoss()


def l2_loss(predicted, target):
    output = loss(predicted, target)

    return output


def train(model, nr_epochs, optimizer, writer, device, dl, test_dl):
    i =0
    for e in range(nr_epochs):
        optimizer.zero_grad()
        i = train_step(model, optimizer, writer,e, device, dl, test_dl, i)


def eval(model,test_ds, writer, i, device):
    loss = 0
    iter = 0
    with torch.no_grad():
        for i_batch, img_test in enumerate(test_ds):
            img_test = img_test.to(device)
            recon_combined, recons, masks, slots = model(img_test)
            loss_value = l2_loss(recon_combined, img_test)
            loss += loss_value
            iter +=10
    writer.add_scalar('Loss test', loss, i)


def plot_grad_flow(named_parameters, writer):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    fig, ax = plt.subplots()
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0010, top=0.6)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    fig.tight_layout()
    writer.add_figure(tag=f'gradient_flow_custom_attention',
                      figure=fig)


def train_step(enc, optimizer, writer, epoch, device, dl, test_ds, i):
    print(epoch)
    accumulated_loss = 0
    accumulation_steps = 0
    for i_batch, sample_batched in enumerate(dl):
        sample_batched = sample_batched.to(device)
        recon_combined, recons, masks, slots = enc(sample_batched)
        loss_value = l2_loss(recon_combined, sample_batched)
        accumulated_loss +=loss_value
        if accumulation_steps == 1:
            accumulated_loss = accumulated_loss/accumulation_steps
            accumulated_loss.backward()
            optimizer.step()
            accumulation_steps = 0
            accumulated_loss = 0
        accumulation_steps+=1


        writer.add_scalar('Loss iter', loss_value, i)
        if i % 10000 == 0 and i != 0:
            eval(enc, test_ds, writer, i, device)
        if i % 500 == 0:
            plot_grad_flow(enc.named_parameters(), writer)
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
    writer = SummaryWriter(f'results/object_discovery_init_acc20_lr0.0001')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    enc = AutoEncSlot()

    if device == torch.device('cuda'):
        enc.to(device)
    optimizer = optim.Adam(enc.parameters(), lr=0.0001, weight_decay=0.5)
    train(enc, 1000, optimizer, writer, device, dl_train,dl_test)


if __name__ == '__main__':
    main()



