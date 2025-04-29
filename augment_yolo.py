import cv2
import albumentations as A
import os
import glob
import argparse


def read_yolo_txt(label_path, img_w, img_h):
    """
    读取 YOLO txt 并转换为 pascal_voc 绝对坐标格式列表。
    跳过格式不正确的行，日志警告后继续。
    返回 [(cls_id, [xmin, ymin, xmax, ymax]), ...]
    """
    objs = []
    if not os.path.exists(label_path):
        return objs
    with open(label_path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            # 确保至少五个字段，超过时只取前五
            if len(parts) < 5:
                print(f"Warning: skipping malformed line {lineno} in {label_path}: '{line.strip()}'")
                continue
            cls_id = parts[0]
            try:
                x_c, y_c, w_n, h_n = map(float, parts[1:5])
            except ValueError:
                print(f"Warning: non-float values on line {lineno} in {label_path}: '{line.strip()}'")
                continue
            # 转为 pascal_voc 绝对坐标
            x_c *= img_w; y_c *= img_h
            w = w_n * img_w; h = h_n * img_h
            xmin = max(0, x_c - w/2)
            ymin = max(0, y_c - h/2)
            xmax = min(img_w, x_c + w/2)
            ymax = min(img_h, y_c + h/2)
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: zero-area box on line {lineno} in {label_path}")
                continue
            objs.append((cls_id, [xmin, ymin, xmax, ymax]))
    return objs


def write_yolo_txt(label_path, objs, img_w, img_h):
    """
    将 pascal_voc 坐标列表写回 YOLO txt 格式，保证每行恰好 5 项。
    输出后若文件为空，则删除以防 LabelImg 解析出错。
    """
    lines = []
    for cls_id, box in objs:
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin; h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        x_c = xmin + w/2; y_c = ymin + h/2
        x_c_n = x_c / img_w; y_c_n = y_c / img_h
        w_n = w / img_w; h_n = h / img_h
        # 保证归一化值在 (0,1]
        if not (0 < x_c_n <= 1 and 0 < y_c_n <= 1 and 0 < w_n <= 1 and 0 < h_n <= 1):
            print(f"Warning: normalized values out of range for box {box} in file {label_path}")
            continue
        lines.append(f"{cls_id} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f}\n")
    if lines:
        with open(label_path, 'w') as f:
            f.writelines(lines)
    else:
        # 删除空文件，避免 LabelImg 加载出错
        if os.path.exists(label_path):
            os.remove(label_path)


def augment_yolo_data(image_dir, label_dir, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 使用 pascal_voc 处理绝对坐标，再转换回 YOLO
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.0)
    transforms = {
        'contrast': A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.1,0.3), p=1)], bbox_params=bbox_params),
        'brightness': A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.1,0.3), contrast_limit=0, p=1)], bbox_params=bbox_params),
        'hflip': A.Compose([A.HorizontalFlip(p=1)], bbox_params=bbox_params),
        'vflip': A.Compose([A.VerticalFlip(p=1)], bbox_params=bbox_params),
        'combo': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.1,0.3), contrast_limit=(0.1,0.3), p=1),
            A.HorizontalFlip(p=0.5)
        ], bbox_params=bbox_params)
    }

    for img_path in glob.glob(os.path.join(image_dir, '*')):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ['.jpg','.jpeg','.png','.bmp']:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(label_dir, base + '.txt')

        objs = read_yolo_txt(txt_path, w, h)
        boxes = [box for _, box in objs]
        labels = [cls for cls, _ in objs]

        for name, aug in transforms.items():
            data = aug(image=img, bboxes=boxes, labels=labels)
            aug_img = data['image']; aug_bboxes = data['bboxes']; aug_labels = data['labels']
            new_img_name = f"{base}_{name}{ext}"
            cv2.imwrite(os.path.join(out_img_dir, new_img_name), aug_img)

            new_lbl_path = os.path.join(out_lbl_dir, base + f"_{name}.txt")
            write_yolo_txt(new_lbl_path, list(zip(aug_labels, aug_bboxes)), w, h)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Augment YOLO formatted images and labels')
    parser.add_argument('--image-dir', default=r"E:\opencvProject\LYJ\LiSheng\images\20250425\Roi\images")
    parser.add_argument('--label-dir', default=r"E:\opencvProject\LYJ\LiSheng\images\20250425\Roi\labels")
    parser.add_argument('--out-img-dir', default=r"E:\opencvProject\LYJ\LiSheng\images\20250425\Roi\images_aug")
    parser.add_argument('--out-lbl-dir', default=r"E:\opencvProject\LYJ\LiSheng\images\20250425\Roi\labels_aug")
    args = parser.parse_args()
    augment_yolo_data(args.image_dir, args.label_dir, args.out_img_dir, args.out_lbl_dir)
