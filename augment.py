import cv2
import albumentations as A
import os
import glob
import argparse
import json


def read_txt_labels(label_path, width, height):
    objs = []
    if not os.path.exists(label_path):
        return objs
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue
            # 已归一化 -> 保持 norm
            pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            objs.append((cls_id, pts))
    return objs


def write_txt_labels(label_path, objs_norm):
    with open(label_path, 'w') as f:
        for cls_id, pts in objs_norm:
            flat = []
            for x, y in pts:
                flat.extend([f"{x:.6f}", f"{y:.6f}"])
            f.write(" ".join([cls_id] + flat) + "\n")


def read_json_labels(label_path, width, height):
    if not os.path.exists(label_path):
        return None, []
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    objs = []
    for shape in data.get('shapes', []):
        cls_id = shape.get('label', '')
        pts = shape.get('points', [])
        norm_pts = [(x / width, y / height) for x, y in pts]
        objs.append((cls_id, norm_pts))
    return data, objs


def write_json_labels(label_path, template, objs_norm, width, height, img_name):
    out = {
        'version': template.get('version'),
        'flags': template.get('flags', {}),
        'shapes': [],
        'imagePath': img_name,
        'imageData': None,
        'imageHeight': height,
        'imageWidth': width
    }
    # 遍历原始 shapes 并对应新的 objs_norm
    for shape, (cls_id, pts_norm) in zip(template.get('shapes', []), objs_norm):
        pts_abs = [(x * width, y * height) for x, y in pts_norm]
        out['shapes'].append({
            'label': cls_id,
            'points': pts_abs,
            'group_id': shape.get('group_id'),
            'shape_type': shape.get('shape_type', 'polygon'),
            'flags': shape.get('flags', {})
        })
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def image_augmentation_with_labels(image_dir, label_dir, img_out_dir, lbl_out_dir, label_format):
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    # 定义单次增强：亮度、对比度、水平翻转、垂直翻转
    single_augs = {
        'contrast': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.1, 0.3), p=1)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
        'brightness': A.Compose(