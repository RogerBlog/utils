import json
import os
import cv2
import numpy as np
from PIL import Image

# 配置参数（根据实际情况修改）
CLASS_MAPPING = {
    "fracture": 255  # 将目标类别设为白色（255为最大可见值）
}
OUTPUT_MASK_DIR = "masks"  # 单独存放mask的目录
DEBUG_MODE = True  # 开启调试输出


def labelme_to_elfin(json_path, img_folder, output_dir):
    """核心转换函数（包含完整错误处理）"""
    try:
        # ================== 1. 数据加载与校验 ==================
        with open(json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)

        # 获取关联图像路径
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = find_image_file(img_folder, base_name)
        if not img_path:
            print(f"⚠️ 未找到 {base_name} 对应的图像文件")
            return

        # ================== 2. 获取真实图像尺寸 ==================
        with Image.open(img_path) as img:
            img_width, img_height = img.size  # 优先使用实际尺寸
            if DEBUG_MODE:
                print(f"\n🔍 图像尺寸验证：")
                print(f"  JSON尺寸：{labelme_data['imageWidth']}x{labelme_data['imageHeight']}")
                print(f"  实际尺寸：{img_width}x{img_height}")

        # ================== 3. 创建空白掩膜 ==================
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # ================== 4. 处理多边形数据 ==================
        valid_shapes = 0
        for shape in labelme_data['shapes']:
            label_name = shape['label']
            if label_name not in CLASS_MAPPING:
                print(f"🚨 发现未配置的标签：{label_name}")
                continue

            # 坐标转换与校验
            points = process_coordinates(shape['points'], img_width, img_height)
            if points is None:
                continue

            # 绘制多边形
            cv2.fillPoly(mask, [points], color=CLASS_MAPPING[label_name])
            valid_shapes += 1

        # ================== 5. 结果验证与保存 ==================
        if valid_shapes == 0:
            print(f"⛔ 文件 {base_name} 无有效标注")
            return

        # 保存掩膜文件
        os.makedirs(os.path.join(output_dir, OUTPUT_MASK_DIR), exist_ok=True)
        mask_path = os.path.join(output_dir, OUTPUT_MASK_DIR, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # 生成精灵标注格式
        elfin_data = build_elfin_data(labelme_data, base_name, mask_path)

        # 保存JSON
        output_path = os.path.join(output_dir, f"{base_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(elfin_data, f, indent=2, ensure_ascii=False)

        # ================== 6. 调试输出 ==================
        if DEBUG_MODE:
            print(f"\n✅ 转换成功：{base_name}")
            print(f"   掩膜路径：{mask_path}")
            print(f"   唯一像素值：{np.unique(mask)}")
            visualize_mask(mask)  # 可视化验证

    except Exception as e:
        print(f"❌ 转换失败：{json_path}")
        print(f"   错误信息：{str(e)}")


def find_image_file(img_folder, base_name):
    """查找关联图像文件（支持中文路径）"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        possible_path = os.path.join(img_folder, f"{base_name}{ext}")
        if os.path.exists(possible_path):
            return possible_path
    return None


def process_coordinates(raw_points, img_width, img_height):
    """坐标处理（带边界校验）"""
    try:
        points = np.array(raw_points, dtype=np.float32)
        points = np.round(points).astype(np.int32)

        # 边界裁剪（防止越界）
        points[:, 0] = np.clip(points[:, 0], 0, img_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img_height - 1)

        if DEBUG_MODE:
            print(f"  坐标范围：X({np.min(points[:, 0])}-{np.max(points[:, 0])}) "
                  f"Y({np.min(points[:, 1])}-{np.max(points[:, 1])})")

        return points
    except Exception as e:
        print(f"  坐标转换失败：{str(e)}")
        return None


def build_elfin_data(labelme_data, base_name, mask_path):
    """构建精灵标注数据结构"""
    return {
        "version": "1.0",
        "imageHeight": labelme_data['imageHeight'],
        "imageWidth": labelme_data['imageWidth'],
        "imageUrl": labelme_data['imagePath'],
        "maskPath": os.path.relpath(mask_path, start=output_dir),
        "labelType": "segmentation",
        "labels": [{
            "class": shape['label'],
            "data": shape['points'],
            "type": "polygon"
        } for shape in labelme_data['shapes'] if shape['label'] in CLASS_MAPPING]
    }


def visualize_mask(mask):
    """掩膜可视化（弹出窗口显示）"""
    cv2.imshow('Mask Preview', mask * 255)  # 将掩膜值映射到0-255范围
    cv2.waitKey(100)  # 显示100ms后自动关闭


def batch_convert(input_dir, img_folder, output_dir):
    """批量转换入口"""
    print(f"🚀 开始批量转换，共发现 {len(os.listdir(input_dir))} 个JSON文件")
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(input_dir, file_name)
            labelme_to_elfin(json_path, img_folder, output_dir)
    print("🎉 批量转换完成！")


if __name__ == "__main__":
    # ======== 用户配置区域 ========
    input_dir = r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\new"  # Labelme JSON文件夹
    img_folder = r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\images"  # 原始图像文件夹
    output_dir = r"C:\Users\Administrator\Desktop\ultralytics-main\runs\segment\20250228\jingling"  # 输出目录

    # ======== 执行转换 ========
    batch_convert(input_dir, img_folder, output_dir)