import cv2
import os
import shutil


def resize_org():
    root_src_dir = "lfw"
    org_dir = "original"
    resize_dir = "resized"
    for root, dirnames, filenames in os.walk(root_src_dir):
        for filename in filenames:
            if filename.endswith(('.jpg')):
                full_path = os.path.join(root, filename)
                org_path = os.path.join(org_dir, filename)
                resize_path = os.path.join(resize_dir, filename)
                img = cv2.imread(full_path)
                img = cv2.resize(img, (125, 125))
                cv2.imwrite(resize_path, img)
                shutil.copy(full_path, org_path)

if __name__ == "__main__":
    resize_org()