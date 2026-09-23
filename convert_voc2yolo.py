import os
import xml.etree.ElementTree as ET
from shutil import copyfile

# ===================== 路径配置（和你的目录完全匹配）=====================
VOC_ROOT = "." # 因为你现在就在 Tongue 文件夹里，所以用 . 代表当前目录
IMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
ANNO_DIR = os.path.join(VOC_ROOT, "Annotations")
TRAIN_TXT = os.path.join(VOC_ROOT, "ImageSets/Main/train.txt")
VAL_TXT = os.path.join(VOC_ROOT, "ImageSets/Main/val.txt")
YOLO_ROOT = "data" # 转换后生成的 YOLO 数据集目录

# 创建 YOLO 所需目录
os.makedirs(f"{YOLO_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/images/val", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/labels/train", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/labels/val", exist_ok=True)

def convert_xml_to_yolo(xml_path, img_w, img_h):
    """将 VOC XML 标注转换为 YOLO 格式的 txt 标签"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_labels = []
    for obj in root.findall("object"):
        cls = 0 # 舌体只有 1 类，类别 ID 固定为 0
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        # 转换为 YOLO 格式：中心点 x, 中心点 y, 宽, 高（均归一化）
        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        yolo_labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return yolo_labels

# ===================== 处理训练集 =====================
with open(TRAIN_TXT, "r", encoding="utf-8") as f:
    train_ids = [line.strip() for line in f if line.strip()]

for img_id in train_ids:
    # 复制图片到 YOLO 训练集目录
    copyfile(
        os.path.join(IMAGES_DIR, f"{img_id}.jpg"),
        os.path.join(YOLO_ROOT, "images/train", f"{img_id}.jpg")
    )
    # 转换 XML 标注为 YOLO txt
    xml_path = os.path.join(ANNO_DIR, f"{img_id}.xml")
    tree = ET.parse(xml_path)
    size = tree.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    labels = convert_xml_to_yolo(xml_path, img_w, img_h)
    with open(os.path.join(YOLO_ROOT, "labels/train", f"{img_id}.txt"), "w") as f:
        f.write("\n".join(labels))

# ===================== 处理验证集 =====================
with open(VAL_TXT, "r", encoding="utf-8") as f:
    val_ids = [line.strip() for line in f if line.strip()]

for img_id in val_ids:
    copyfile(
        os.path.join(IMAGES_DIR, f"{img_id}.jpg"),
        os.path.join(YOLO_ROOT, "images/val", f"{img_id}.jpg")
    )
    xml_path = os.path.join(ANNO_DIR, f"{img_id}.xml")
    tree = ET.parse(xml_path)
    size = tree.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    labels = convert_xml_to_yolo(xml_path, img_w, img_h)
    with open(os.path.join(YOLO_ROOT, "labels/val", f"{img_id}.txt"), "w") as f:
        f.write("\n".join(labels))

print("✅ VOC 数据集已成功转换为 YOLO 格式！")
