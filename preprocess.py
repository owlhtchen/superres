import cv2
import os
import shutil
import torch
import torch.nn.functional as F

def resize_org():
    root_src_dir = "lfw"
    org_dir = "original_80"
    resize_dir = "resized_40"
    for root, dirnames, filenames in os.walk(root_src_dir):
        for filename in filenames:
            if filename.endswith(('.jpg')):
                full_path = os.path.join(root, filename)
                org_path = os.path.join(org_dir, filename)
                resize_path = os.path.join(resize_dir, filename)
                original = cv2.imread(full_path)
                img = cv2.resize(original, dsize=(40, 40), interpolation=cv2.INTER_AREA)
                cv2.imwrite(resize_path, img)
                img = cv2.resize(original, dsize=(80, 80), interpolation=cv2.INTER_AREA)
                cv2.imwrite(org_path, img)

if __name__ == "__main__":
    resize_org()