import os
import shutil
import datetime

# 源文件夹（包含图像和标签文件，例如 .json 文件）
source_folder = r'E:\opencvProject\LYJ\LiSheng\images\20250221\20250221_roi\images_copy'
# 目标文件夹（将复制后的图像与标签都存放在此文件夹）
dest_folder = r'E:\opencvProject\LYJ\LiSheng\images\20250221\20250221_roi\images2'

os.makedirs(dest_folder, exist_ok=True)

# 获取当前时间并格式化（作为新文件名的一部分）
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
counter = 1

# 支持的图片扩展名列表
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']

for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    if not os.path.isfile(file_path):
        continue

    ext = os.path.splitext(filename)[1].lower()
    if ext in image_extensions:
        # 根据当前时间和计数器生成新的文件名（保持原图片扩展名）
        new_base = f"{current_time}_{counter}"
        new_image_name = new_base + ext
        dest_image_path = os.path.join(dest_folder, new_image_name)

        # 复制图像文件到目标文件夹
        shutil.copyfile(file_path, dest_image_path)
        print(f"图像复制成功：{filename} -> {new_image_name}")

        # 假设标签文件与图像同名，扩展名为 .json
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_file_path = os.path.join(source_folder, label_filename)
        if os.path.exists(label_file_path):
            # 标签文件用和图像文件相同的命名规则，新扩展名固定为 .json
            new_label_name = new_base + '.txt'
            dest_label_path = os.path.join(dest_folder, new_label_name)
            shutil.copyfile(label_file_path, dest_label_path)
            print(f"标签复制成功：{label_filename} -> {new_label_name}")
        else:
            print(f"未找到标签文件：{label_filename}")

        counter += 1
