from audioop import bias
from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self):
        self.input_dir = "resized"
        self.gt_dir = "original"
        input_names = set(os.listdir(self.input_dir))
        gt_names = set(os.listdir(self.gt_dir))
        assert(input_names == gt_names)
        self.image_names = list(input_names)[:900]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.input_dir, image_name)
        input_image = cv2.imread(image_path)
        bicubic_input = cv2.resize(input_image, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC).astype('float32') / 255.0
        bicubic_input = cv2.cvtColor(bicubic_input, cv2.COLOR_BGR2YCR_CB)
        input_image = input_image.astype('float32') / 255.0
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCR_CB)
        gt_image = cv2.imread(os.path.join(self.gt_dir, image_name)).astype('float32') / 255.0
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCR_CB)
        input_image = torch.tensor(input_image, device=device).permute(2, 0, 1).float()
        bicubic_input = torch.tensor(bicubic_input, device=device).permute(2, 0, 1).float()
        gt_image = torch.tensor(gt_image, device=device).permute(2, 0, 1).float()
        return input_image, bicubic_input, gt_image, image_name

class SuperRes(nn.Module):
    def __init__(self):
        super().__init__()
        # self.feature_ch = [96, 76, 65, 55, 47, 39, 32]
        # self.reconstruct_ch = {"A1": 64, "B1": 32, "B2": 32, "L": 4}
        self.scale_factor = 2
        self.feature_ch = [32, 26, 22, 18, 14, 11, 8]
        self.reconstruct_ch = {"A1": 24, "B1": 8, "B2": 8, "L": 4}
        self.conv_cnn0 = nn.Conv2d(1, self.feature_ch[0], kernel_size=3, padding=1, bias=True)
        self.conv_cnn1 = nn.Conv2d(self.feature_ch[0], self.feature_ch[1], kernel_size=3, padding=1, bias=True)
        self.conv_cnn2 = nn.Conv2d(self.feature_ch[1], self.feature_ch[2], kernel_size=3, padding=1, bias=True)
        self.conv_cnn3 = nn.Conv2d(self.feature_ch[2], self.feature_ch[3], kernel_size=3, padding=1, bias=True)
        self.conv_cnn4 = nn.Conv2d(self.feature_ch[3], self.feature_ch[4], kernel_size=3, padding=1, bias=True)
        self.conv_cnn5 = nn.Conv2d(self.feature_ch[4], self.feature_ch[5], kernel_size=3, padding=1, bias=True)
        self.conv_cnn6 = nn.Conv2d(self.feature_ch[5], self.feature_ch[6], kernel_size=3, padding=1, bias=True)
        self.cnn0 = nn.Sequential(self.conv_cnn0, nn.PReLU())
        self.cnn1 = nn.Sequential(self.conv_cnn1, nn.PReLU())
        self.cnn2 = nn.Sequential(self.conv_cnn2, nn.PReLU())
        self.cnn3 = nn.Sequential(self.conv_cnn3, nn.PReLU())
        self.cnn4 = nn.Sequential(self.conv_cnn4, nn.PReLU())
        self.cnn5 = nn.Sequential(self.conv_cnn5, nn.PReLU())
        self.cnn6 = nn.Sequential(self.conv_cnn6, nn.PReLU())
        self.concat_ch_1 = sum(self.feature_ch)
        self.cnn_A1 = nn.Conv2d(self.concat_ch_1, self.reconstruct_ch['A1'], kernel_size=1, padding=0, bias=True)
        self.cnn_B1 = nn.Conv2d(self.concat_ch_1, self.reconstruct_ch['B1'], kernel_size=1, padding=0, bias=True)
        self.cnn_B2 = nn.Conv2d(self.reconstruct_ch['B1'], self.reconstruct_ch['B2'], kernel_size=3, padding=1, bias=True)
        self.concat_ch_2 = self.reconstruct_ch['A1'] + self.reconstruct_ch['B2']
        self.A1 = nn.Sequential(
            self.cnn_A1,
            nn.PReLU()
        )
        self.B = nn.Sequential(
            self.cnn_B1,
            nn.PReLU(),
            self.cnn_B2,
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
        # out_r1 = torch.concat((output[:, 0:1, :, :], output[:, 1:2, :, :]), dim=2)
        # out_r2 = torch.concat((output[:, 2:3, :, :], output[:, 3:4, :, :]), dim=2)
        # return torch.concat((out_r1, out_r2), dim=3)
        out_tensor = torch.zeros((n, c, h * self.scale_factor, w * self.scale_factor), device=device)
        # out_tensor[:, :, torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor)] = output[:, 0:1, :, :]
        # out_tensor[:, :, torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor)] = output[:, 1:2, :, :]
        # out_tensor[:, :, torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor)] = output[:, 2:3, :, :]
        # out_tensor[:, :, torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor)] = output[:, 3:4, :, :]
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x, grid_y] = output[:, 0:1, :, :]
        grid_x, grid_y = torch.meshgrid(torch.arange(0, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x, grid_y] = output[:, 1:2, :, :]
        grid_x, grid_y = torch.meshgrid(torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(0, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x, grid_y] = output[:, 2:3, :, :]
        grid_x, grid_y = torch.meshgrid(torch.arange(1, h * self.scale_factor, self.scale_factor), torch.arange(1, w * self.scale_factor, self.scale_factor), indexing='ij')
        out_tensor[:, :, grid_x, grid_y] = output[:, 3:4, :, :]
        return out_tensor

if __name__ == "__main__":
    train_dataset = CustomImageDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=False)
    # test_input, test_bicubic_input, test_gt = next(iter(train_dataset))
    # print(test_bicubic_input.shape)
    model = SuperRes()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 50
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_image, bicubic_input, gt_image, image_name = data
            # print("bicubic_input.shape={}".format(bicubic_input.shape))
            output_y = model(input_image[:, 0:1, :, :]) # only Y-channel
            print("output_y.shape={}".format(output_y.shape))
            # print("input_image[:, 1:, :, :].shape={}".format(bicubic_input[:, 1:, :, :].shape))
            new_images = torch.concat([output_y, bicubic_input[:, 1:, :, :]], dim=1)
            # new_images = torch.concat([bicubic_input[:, :1, :, :], bicubic_input[:, 1:, :, :]], dim=1) # ok
            # print("new_images.shape={}".format(new_images.shape))
            loss = loss_fn(output_y, gt_image[:, 0:1, :, :])
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            print("batch={}".format(i))
        one_image = new_images[0, :, :, :].cpu().permute(1, 2, 0).detach().numpy()
        # one_image = bicubic_input[0, :, :, :].cpu().permute(1, 2, 0).detach().numpy() # ok
        one_image = (cv2.cvtColor(one_image, cv2.COLOR_YCrCb2BGR) * 255.0).astype('uint8')
        print("one_image.shape={}".format(one_image.shape))
        print("image_path={}".format(image_name[0]))
        debug_out_path = os.path.join("debug", image_name[0])
        cv2.imwrite(debug_out_path, one_image)        
        print("epoch={}, loss={}".format(epoch, epoch_loss))
