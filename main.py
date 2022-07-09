import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class CustomImageDataset(Dataset):
    def __init__(self):
        self.input_dir = "resized"
        self.gt_dir = "original"
        input_names = set(os.listdir(self.input_dir))
        gt_names = set(os.listdir(self.gt_dir))
        assert(input_names == gt_names)
        self.image_names = list(input_names)[:900]
        self.images = [self.get(i) for i in tqdm(range(len(self)))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        return self.images[idx]

    def get(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.input_dir, image_name)
        input_image = cv2.imread(image_path)
        bicubic_input = np.clip(cv2.resize(input_image, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC).astype('float32') / 255.0, 0.0, 1.0)
        bicubic_input = np.clip(cv2.cvtColor(bicubic_input, cv2.COLOR_BGR2YCR_CB), 0.0, 1.0)
        input_image = input_image.astype('float32') / 255.0
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCR_CB)
        gt_image = cv2.imread(os.path.join(self.gt_dir, image_name)).astype('float32') / 255.0
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCR_CB)
        input_image = torch.tensor(input_image).permute(2, 0, 1).float().contiguous()
        bicubic_input = torch.tensor(bicubic_input).permute(2, 0, 1).float().contiguous()
        gt_image = torch.tensor(gt_image).permute(2, 0, 1).float().contiguous()
        return input_image, gt_image, image_name, bicubic_input

class SuperRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_ch = [96, 76, 65, 55, 47, 39, 32]
        self.reconstruct_ch = {"A1": 64, "B1": 32, "B2": 32, "L": 4}
        self.scale_factor = 2
        # self.feature_ch = [32, 26, 22, 18, 14, 11, 8]
        # self.reconstruct_ch = {"A1": 24, "B1": 8, "B2": 8, "L": 4}
        conv_cnn0 = nn.Conv2d(1, self.feature_ch[0], kernel_size=3, padding=1, bias=True)
        conv_cnn1 = nn.Conv2d(self.feature_ch[0], self.feature_ch[1], kernel_size=3, padding=1, bias=True)
        conv_cnn2 = nn.Conv2d(self.feature_ch[1], self.feature_ch[2], kernel_size=3, padding=1, bias=True)
        conv_cnn3 = nn.Conv2d(self.feature_ch[2], self.feature_ch[3], kernel_size=3, padding=1, bias=True)
        conv_cnn4 = nn.Conv2d(self.feature_ch[3], self.feature_ch[4], kernel_size=3, padding=1, bias=True)
        conv_cnn5 = nn.Conv2d(self.feature_ch[4], self.feature_ch[5], kernel_size=3, padding=1, bias=True)
        conv_cnn6 = nn.Conv2d(self.feature_ch[5], self.feature_ch[6], kernel_size=3, padding=1, bias=True)
        self.cnn0 = nn.Sequential(conv_cnn0, nn.PReLU())
        self.cnn1 = nn.Sequential(conv_cnn1, nn.PReLU())
        self.cnn2 = nn.Sequential(conv_cnn2, nn.PReLU())
        self.cnn3 = nn.Sequential(conv_cnn3, nn.PReLU())
        self.cnn4 = nn.Sequential(conv_cnn4, nn.PReLU())
        self.cnn5 = nn.Sequential(conv_cnn5, nn.PReLU())
        self.cnn6 = nn.Sequential(conv_cnn6, nn.PReLU())
        self.concat_ch_1 = sum(self.feature_ch)
        cnn_A1 = nn.Conv2d(self.concat_ch_1, self.reconstruct_ch['A1'], kernel_size=1, padding=0, bias=True)
        cnn_B1 = nn.Conv2d(self.concat_ch_1, self.reconstruct_ch['B1'], kernel_size=1, padding=0, bias=True)
        cnn_B2 = nn.Conv2d(self.reconstruct_ch['B1'], self.reconstruct_ch['B2'], kernel_size=3, padding=1, bias=True)
        self.concat_ch_2 = self.reconstruct_ch['A1'] + self.reconstruct_ch['B2']
        self.A1 = nn.Sequential(
            cnn_A1,
            nn.PReLU()
        )
        self.B = nn.Sequential(
            cnn_B1,
            nn.PReLU(),
            cnn_B2,
            nn.PReLU()
        )
        self.L = nn.Conv2d(self.reconstruct_ch['A1'] + self.reconstruct_ch['B2'], self.scale_factor * self.scale_factor, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        n, c, h, w = x.shape
        x0 = self.cnn0(x)
        x1 = self.cnn1(x0)
        x2 = self.cnn2(x1)
        x3 = self.cnn3(x2)
        x4 = self.cnn4(x3)
        x5 = self.cnn5(x4)
        x6 = self.cnn6(x5)
        feature_concat = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=1)
        a_out = self.A1(feature_concat)
        b_out = self.B(feature_concat)
        reconstruct_concat = torch.cat([a_out, b_out], dim=1)
        output = self.L(reconstruct_concat)
        # print("c={}".format(c))
        out_tensor = torch.zeros((n, 1, h * self.scale_factor, w * self.scale_factor), device=device)
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor), indexing='ij')
        # print("grid_x={}".format(grid_x))
        # print("grid_y={}".format(grid_y))
        out_tensor[:, :, grid_x, grid_y] = output[:, 0:1, :, :].contiguous()
        # grid_x, grid_y = torch.meshgrid(torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x + 1, grid_y] = output[:, 1:2, :, :].contiguous()
        # grid_x, grid_y = torch.meshgrid(torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x, grid_y + 1] = output[:, 2:3, :, :].contiguous()
        # grid_x, grid_y = torch.meshgrid(torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x + 1, grid_y + 1] = output[:, 3:4, :, :].contiguous()
        return out_tensor.contiguous()

if __name__ == "__main__":
    train_dataset = CustomImageDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=False)
    # test_input, test_bicubic_input, test_gt = next(iter(train_dataset))
    # print(test_bicubic_input.shape)
    model = SuperRes()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_image, gt_image, image_name, bicubic_input = data
            input_image = input_image.to(device)
            bicubic_input = bicubic_input.to(device)
            gt_image = gt_image.to(device)
            # print("bicubic_input.shape={}".format(bicubic_input.shape))
            output_y = model(input_image[:, 0:1, :, :]) # only Y-channel
            # print("output_y.shape={}".format(output_y.shape))
            # print("input_image[:, 1:, :, :].shape={}".format(bicubic_input[:, 1:, :, :].shape))
            new_images = torch.concat([output_y, bicubic_input[:, 1:, :, :]], dim=1)
            # new_images = torch.concat([bicubic_input[:, :1, :, :], bicubic_input[:, 1:, :, :]], dim=1) # ok
            # print("new_images.shape={}".format(new_images.shape))
            loss = loss_fn(output_y, gt_image[:, 0:1, :, :])
            epoch_loss += float(loss.item())
            loss.backward()
            optimizer.step()
            print("batch={}, loss={}".format(i, loss.item()))
        one_image = np.clip(new_images[0, :, :, :].cpu().permute(1, 2, 0).detach().numpy(), 0.0, 1.0)
        assert not np.any(np.isnan(one_image))
        # one_image = bicubic_input[0, :, :, :].cpu().permute(1, 2, 0).detach().numpy() # ok
        # cv2.imwrite('debug/one_image_ycc{}.exr'.format(epoch), one_image)
        # cv2.imwrite('debug/one_image{}.exr'.format(epoch), cv2.cvtColor(one_image, cv2.COLOR_YCrCb2BGR))
        one_image = (cv2.cvtColor(one_image, cv2.COLOR_YCrCb2BGR) * 255.0)#.astype('uint8') # FIX weird dots astype('uint8') can make negative values become a large positive value
        print("one_image.shape={}".format(one_image.shape))
        print("image_path={}".format(image_name[0]))
        debug_out_path = os.path.join("debug", image_name[0])
        cv2.imwrite(debug_out_path, one_image)        
        print("epoch={}, loss={}".format(epoch, epoch_loss))
