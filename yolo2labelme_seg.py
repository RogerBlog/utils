import json
import os
from PIL import Image

# 反向类别映射表（根据你的原始映射修改）
class_id_to_label = {
    0: "fracture",
    # 添加更多反向映射（例如 1: "other_class"）
}

def convert_yolo_txt_to_labelme(txt_folder, img_folder, output_folder):
    """
    参数说明：
    txt_folder: YOLO格式txt文件夹路径
    img_folder: 对应的图像文件夹路径
    output_folder: JSON文件输出路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有txt文件
    for txt_name in os.listdir(txt_folder):
        if not txt_name.endswith(".txt"):
            continue

        # 获取对应的图像路径
        base_name = os.path.splitext(txt_name)[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            possible_path = os.path.join(img_folder, f"{base_name}{ext}")
            if os.path.exists(possible_path):
                img_path = possible_path
                break

        if not img_path:
            print(f"警告：找不到 {base_name} 对应的图像文件")
            continue

        # 获取图像尺寸
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # 准备Labelme数据结构
        labelme_data = {
            "version": "5.6.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        # 读取txt文件内容
        txt_path = os.path.join(txt_folder, txt_name)
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # 解析每个检测结果
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3 or len(parts) % 2 != 1:
                print(f"无效行格式：{line}")
                continue

            # 解析类别ID和坐标
            class_id = int(parts[0])
            normalized_points = list(map(float, parts[1:]))

            # 检查坐标数量有效性
            if len(normalized_points) % 2 != 0:
                print(f"坐标数量不正确：{line}")
                continue

            # 获取类别标签
            label = class_id_to_label.get(class_id)
            if label is None:
                print(f"警告：类别ID {class_id} 未在映射表中")
                continue

            # 转换为绝对坐标
            points = []
            for i in range(0, len(normalized_points), 2):
                x = normalized_points[i] * img_width
                y = normalized_points[i+1] * img_height
                points.append([x, y])

            # 添加shape信息
            labelme_data["shapes"].append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

        # 保存JSON文件
        output_path = os.path.join(output_folder, f"{base_name}.json")
        with open(output_path, "w") as f:
            json.dump(labelme_data, f, indent=2)
        print(f"已转换：{txt_name} -> {os.path.basename(output_path)}")

# 使用示例
if __name__ == "__main__":
    convert_yolo_txt_to_labelme(
        txt_folder=r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\labels",
        img_folder=r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\images",
        output_folder=r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\json"
    )