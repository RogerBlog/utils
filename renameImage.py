import cv2
import os



image_path = r"E:\opencvProject\LYJ\LiSheng\images\20250425\Pin\images"
output_path = r"E:\opencvProject\LYJ\LiSheng\images\20250425\Pin\1"

for image_name in os.listdir(image_path):
    image_new_name = image_name.split(".")[0]
    image = cv2.imread(os.path.join(image_path, image_name))

    cv2.imwrite(os.path.join(output_path, image_new_name + ".bmp"), image)


