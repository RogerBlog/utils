import os
from PIL import Image
import datetime

# 设置图像文件夹路径
folder_path = r'E:\opencvProject\LYJ\YongSheng\images\20250307\img'
output_path = r'E:\opencvProject\LYJ\YongSheng\images\20250307\img_'

# 获取当前时间并格式化
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 支持的图片扩展名列表
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']

# 初始化计数器
counter = 1

# 创建输出文件夹（可选）
output_folder = os.path.join(folder_path, output_path)
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 跳过子目录和非文件项
    if not os.path.isfile(file_path):
        continue

    # 检查文件扩展名
    ext = os.path.splitext(filename)[1].lower()
    if ext in image_extensions:
        try:
            # 打开图像文件
            with Image.open(file_path) as img:
                # 处理图像模式
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')

                # 生成新文件名
                new_filename = f"{current_time}_{counter}.png"
                save_path = os.path.join(output_folder, new_filename)

                # 保存图像
                img.save(save_path, 'PNG')
                print(f"成功转换：{filename} -> {new_filename}")
                counter += 1

        except Exception as e:
            print(f"处理 {filename} 时出错：{str(e)}")