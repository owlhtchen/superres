import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output = torch.zeros([1, 4, 2, 2], device=device) # change to [1, 1, 4, 4]
n, c, h, w = output.shape # shape of output
scale_factor = 2
output[:, 0:1, :, :] = 0
output[:, 1:2, :, :] = 1
output[:, 2:3, :, :] = 2
output[:, 3:4, :, :] = 3
print(output)
print("output.shape={}".format(output.shape))

grid_x, grid_y = torch.meshgrid(torch.arange(0, h * scale_factor, scale_factor), torch.arange(0, w * scale_factor, scale_factor), indexing='ij')
out_tensor = torch.zeros((n, 1, h * scale_factor, w * scale_factor), device=device)

out_tensor[:, :, grid_x, grid_y] = output[:, 0:1, :, :].contiguous()
# grid_x, grid_y = torch.meshgrid(torch.arange(0, h * scale_factor, scale_factor), torch.arange(1, w * scale_factor, scale_factor), indexing='ij')
out_tensor[:, :, grid_x + 1, grid_y] = output[:, 1:2, :, :].contiguous()
# grid_x, grid_y = torch.meshgrid(torch.arange(1, h * scale_factor, scale_factor), torch.arange(0, w * scale_factor, scale_factor), indexing='ij')
out_tensor[:, :, grid_x, grid_y + 1] = output[:, 2:3, :, :].contiguous()
# grid_x, grid_y = torch.meshgrid(torch.arange(1, h * scale_factor, scale_factor), torch.arange(1, w * scale_factor, scale_factor), indexing='ij')
out_tensor[:, :, grid_x + 1, grid_y + 1] = output[:, 3:4, :, :].contiguous()
out_tensor.contiguous()
print(out_tensor)
print(out_tensor.shape)