import cv2
import albumentations as A
import os
import glob


def image_augmentation(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义单个增强操作（每个操作独立）
    contrast_aug = A.Compose([  # 随机对比度
        A.RandomBrightnessContrast(
            brightness_limit=0,  # 禁用亮度调整
            contrast_limit=(0.2, 0.5),  # 对比度调整范围
            p=1  # 保证每次都会执行
        )])

    brightness_aug = A.Compose([  # 随机亮度
        A.RandomBrightnessContrast(
            brightness_limit=(0.2, 0.5),
            contrast_limit=0,  # 禁用对比度调整
            p=1
        )])

    flip_aug = A.Compose([  # 水平翻转
        A.HorizontalFlip(p=1)  # 保证每次都会翻转
    ])

    # 定义组合增强操作（包含所有步骤）
    combo_aug = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(0.2, 0.5),
        contrast_limit=(0.2, 0.5),
        p=1
    ),
    A.HorizontalFlip(p=0.5)  # 50%概率翻转

    ])

    # 获取所有图片格式
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    for img_path in image_paths:
    # 读取图片并转换颜色空间
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations使用RGB格式

        # 提取文件名信息
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        ext = ".png"

        # 执行并保存单个增强操作
        # 1. 随机对比度
        augmented = contrast_aug(image=image)['image']
        save_path = os.path.join(output_dir, f"{name}_contrast{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        # 2. 随机亮度
        augmented = brightness_aug(image=image)['image']
        save_path = os.path.join(output_dir, f"{name}_brightness{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        # 3. 水平翻转
        augmented = flip_aug(image=image)['image']
        save_path = os.path.join(output_dir, f"{name}_flip{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        # 4. 组合增强
        augmented = combo_aug(image=image)['image']
        save_path = os.path.join(output_dir, f"{name}_combo{ext}")
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    input_directory = r"E:\opencvProject\LYJ\LFS\images"  # 替换为你的输入目录
    output_directory = r"E:\opencvProject\LYJ\LFS\images_aug"  # 替换为你的输出目录
    image_augmentation(input_directory, output_directory)