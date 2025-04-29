import cv2
import json
import base64
from pathlib import Path
import numpy as np

def mask_to_labelme_json(
    image_path: Path,      # 原始彩色图像（用于尺寸 & Base64 编码）
    mask_path: Path,       # 二值化 mask 图像
    output_json: Path      # 输出 JSON 路径
):
    # --- 1. 读取原图获取尺寸 & 编码 ---
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # --- 2. 读取 mask 并提取轮廓 ---
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # 二值化 :contentReference[oaicite:10]{index=10}
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # 提取外部轮廓 :contentReference[oaicite:11]{index=11}

    # --- 3. 构建 shapes 列表 ---
    shapes = []
    for cnt in contours:
        pts = cnt.squeeze().tolist()
        if len(pts) < 3:  # 至少 3 个点才能成多边形
            continue
        # 如果只有一维列表（特殊情况），封装成多维
        if isinstance(pts[0][0], (int, float)):
            pts = [pts]
        for poly in pts:
            # 构造单个多边形标注项
            shapes.append({
                "label": "line",               # 自定义类别名
                "points": poly,                  # [[x1,y1], [x2,y2], ...]
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })  # shape 格式 :contentReference[oaicite:12]{index=12}

    # --- 4. 组装最终 JSON ---
    labelme_json = {
        "version": "4.6.0",                  # 与 LabelMe 工具版本保持一致 :contentReference[oaicite:13]{index=13}
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w
    }

    # --- 5. 保存 ---
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(labelme_json, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存为 LabelMe 格式：{output_json}")

# === 示例调用 ===
if __name__ == "__main__":
    mask_to_labelme_json(
        image_path=Path(r"E:\opencvProject\LYJ\SaiYiFa\images\20250419\20250419_110748_5.bmp"),
        mask_path=Path(r"E:\opencvProject\LYJ\SaiYiFa\images\20250419\outputs\attachments\20250419_110748_5_1.png"),
        output_json=Path(r"E:\opencvProject\LYJ\SaiYiFa\images\20250419\outputs\attachments\20250419_110748_5.json")
    )
