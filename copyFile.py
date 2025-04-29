import cv2
import os
import numpy as np
import shutil

image_old_file = r"E:\opencvProject\LYJ\LiSheng\images\20250417\images"
image_new_file = r"E:\opencvProject\LYJ\LiSheng\images\20250417\images_copy"
labels_old_file = r"E:\opencvProject\LYJ\LiSheng\images\20250417\labels"
labels_new_file = r"E:\opencvProject\LYJ\LiSheng\images\20250417\labels_copy"


for image_path in os.listdir(image_old_file):
    for labels_path in os.listdir(labels_old_file):
        image_ = image_path.split(".")[0]
        labels_ = labels_path.split(".")[0]
        ori_image_path = os.path.join(image_old_file, image_path)
        new_image_path = os.path.join(image_new_file, image_path)
        ori_labels_path = os.path.join(labels_old_file, labels_path)
        new_labels_path = os.path.join(labels_new_file, labels_path)
        if(image_ == labels_):
            shutil.copyfile(ori_image_path, new_image_path)
            shutil.copyfile(ori_labels_path, new_labels_path)




