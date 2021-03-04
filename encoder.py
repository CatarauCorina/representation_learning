import torch
import math
from torch import nn
from torch.functional import F
import numpy as np
from slot_attention import SlotAttention


def spatial_flatten_channels(x):
    return torch.reshape(x,(-1,x.shape[1],x.shape[2]*x.shape[3]))

def flatten_by_stacking_maps(x):
    return torch.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[3]))


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, resolution):
        super(PositionalEncoding, self).__init__()

        self.dense = nn.Linear(4,hidden_size, bias=True)
        self.grid = self.build_grid(resolution)

    def build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return np.concatenate([grid, 1.0 - grid], axis=-1)

    def forward(self, x):
        positional_vector = self.dense(torch.tensor(self.grid)).permute(0,3,2,1)
        x = x + positional_vector

        return x


class EncoderSlot(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)

    def forward(self, inputs):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        return out


class DecoderSlot(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels=66, out_channels=64, kernel_size=5, stride=(2, 2))
        self.conv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=(2,2))
        self.conv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=(2,2))
        self.conv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=(2,2))
        self.conv_5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2)
        self.conv_6 = nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=2)

    def forward(self, inputs):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        return out


class AutoEncSlot(nn.Module):
    def __init__(self,
                 channels=64,
                 resolution=(112, 112)
                 ):
        super().__init__()
        self.resolution = resolution
        self.decoder_initial_size = (8, 8)
        self.encoder_slot = EncoderSlot()
        self.decoder_slot = DecoderSlot()
        self.pos_enc = PositionalEncoding(channels, resolution)
        self.pos_dec = PositionalEncoding(channels+2, self.decoder_initial_size)
        self.layer_norm = torch.nn.LayerNorm([channels, resolution[0]*resolution[1]])
        self.mlp = nn.Linear(resolution[0]*resolution[1], 64)
        self.slot_attention_model = SlotAttention(num_slots=11, dim=64, iters=3)

        return

    def spatial_flatten_channels(self, x):
        return torch.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3]))

    def sb_decoder(self, z, resolution):
        batch_size = z.shape[0]
        x = torch.linspace(-1, 1, resolution[0])
        y = torch.linspace(-1, 1, resolution[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        z = torch.reshape(z, [-1, z.shape[-1]])[:, :, None, None]
        z = z.expand(-1, -1, resolution[0], resolution[1])
        x_grid = x_grid.expand(11, 1, -1, -1)
        y_grid = y_grid.expand(11, 1, -1, -1)
        x = torch.cat((x_grid, y_grid, z), dim=1)
        return x

    def unstack_and_split(self, x, batch_size, num_channels=3):
        """Unstack batch dimension and split into channels and alpha mask."""
        unstacked = torch.reshape(x, [batch_size, -1] + list(x.shape)[1:])
        channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
        return channels, masks

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.encoder_slot(x)
        x = self.pos_enc(x)
        x = self.spatial_flatten_channels(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        slots = self.slot_attention_model(x)
        x = self.sb_decoder(slots,self.decoder_initial_size)
        x = self.pos_dec(x)
        x = self.decoder_slot(x)
        recons, masks = self.unstack_and_split(x, batch_size=batch_size)
        softmax = torch.nn.Softmax(dim=1)
        masks = softmax(masks)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks, slots

