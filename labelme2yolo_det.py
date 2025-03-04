import json
import os

# 类别映射字典
CLASS_MAPPING = {
    'in': 0,
    'center': 1,
    'out': 2
}


def convert_labelme_to_yolo(json_file_path, output_dir):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 获取图像尺寸
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    # 准备输出路径
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(txt_path, 'w') as txt_file:
        for shape in data['shapes']:
            label = shape['label']
            # 跳过未识别的类别
            if label not in CLASS_MAPPING:
                continue
            class_id = CLASS_MAPPING[label]

            points = shape['points']
            shape_type = shape.get('shape_type', 'rectangle')

            # 计算边界框
            x_min, y_min, x_max, y_max = 0.0, 0.0, 0.0, 0.0

            if len(points) != 2:
                continue
            # 提取并限制坐标范围
            x1 = max(0.0, min(points[0][0], image_width))
            y1 = max(0.0, min(points[0][1], image_height))
            x2 = max(0.0, min(points[1][0], image_width))
            y2 = max(0.0, min(points[1][1], image_height))
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])


            # 计算中心点和宽高
            width = x_max - x_min
            height = y_max - y_min
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            # 归一化
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            # 写入TXT文件
            txt_file.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")


# 批量转换所有JSON文件
def batch_convert(json_dir, output_dir):
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            convert_labelme_to_yolo(json_path, output_dir)


# 使用示例
json_folder = r'E:\opencvProject\LYJ\LFS\images\20250304\labels_json'
output_folder = r'E:\opencvProject\LYJ\LFS\images\20250304\labels'
batch_convert(json_folder, output_folder)