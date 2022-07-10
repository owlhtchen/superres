import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from main import SuperRes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
test_dir = "test_images"
def upsample(img):
    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img = torch.tensor(img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        y = img[:,[0],:,:]
        # img = img.permute(1, 0, 2, 3)
        upsampled_cc = F.interpolate(img[:,1:,:,:], scale_factor=2, mode='bicubic')
        upsampled_y = model(y)
        upsampled = torch.cat([upsampled_y, upsampled_cc], dim=1).squeeze().permute(1,2,0).cpu().numpy()
        # upsampled = F.interpolate(img, scale_factor=2, mode='bicubic').squeeze().permute(1, 2, 0).cpu().numpy()
        # upsampled = model(img).squeeze().permute(1, 2, 0).cpu().numpy()
        upsampled = cv2.cvtColor(upsampled, cv2.COLOR_YCR_CB2BGR)
        return upsampled
def upsample_naive(img):
    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img = torch.tensor(img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        upsampled = F.interpolate(img, scale_factor=2, mode='bicubic').squeeze().permute(1,2,0).cpu().numpy()
        upsampled = cv2.cvtColor(upsampled, cv2.COLOR_YCR_CB2BGR)
        return upsampled


if __name__ == "__main__":
    PT_PATH = "saved.pt"
    model = torch.load(PT_PATH)
    custom_img = [
        cv2.imread("test_images/green-maple-leaf.jpg"),
        cv2.imread("test_images/home-office.jpg"),
        cv2.imread("test_images/martin-luther-king.jpg"),
        cv2.imread("test_images/mount-rushmore.jpg"),
        cv2.imread("test_images/salisbury-cathedral.jpg")]
    downsampled =[cv2.resize(img.astype(np.float32)/255.0, (40,40)) for img in custom_img ]
    upsampled = [upsample(img) for img in downsampled]
    upsampled_naive = [upsample_naive(img) for img in downsampled]
    n = 5
    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(downsampled[i][:,:,::-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(upsampled[i][:,:,::-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(upsampled_naive[i][:,:,::-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig('inference.png')
    