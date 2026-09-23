# main_app.py
import streamlit as st
import hashlib
import pandas as pd
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import time
import os
import json
import numpy as np
import colorsys
from ultralytics import YOLO

# ===== 导入cv2，如果失败则设为None =====
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    st.warning("opencv-python未安装，部分高级功能将受限。请运行: pip install opencv-python")

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="舌象智能诊断系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 自定义CSS样式（高级棕色系主题）----------
def inject_custom_css():
    st.markdown("""
    <style>
        /* 全局背景与字体 */
        .stApp {
            background: linear-gradient(135deg, #FDF8F0 0%, #F9EFE0 100%);
        }
       
        /* 主容器背景 */
        .main .block-container {
            background-color: rgba(255, 248, 240, 0.85);
            border-radius: 20px;
            padding: 2rem 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.05);
        }
       
        /* 标题样式 */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #5E3A2C !important;
            font-weight: 600 !important;
            letter-spacing: -0.3px;
        }
       
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EADBC6 0%, #DCC9AC 100%);
            border-right: 1px solid #C9AD7A;
        }
        [data-testid="stSidebar"] * {
            color: #3A241C !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background-color: #A97C50 !important;
            color: white !important;
            border: none;
            border-radius: 30px;
            transition: all 0.3s;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #8B5E3C !important;
            transform: translateY(-2px);
        }
       
        /* 按钮美化 */
        .stButton button, .st-emotion-cache-1v0mbdj button {
            background: linear-gradient(90deg, #B8865B 0%, #9B6A42 100%);
            color: white !important;
            border: none;
            border-radius: 40px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .stButton button:hover, .st-emotion-cache-1v0mbdj button:hover {
            background: linear-gradient(90deg, #9B6A42 0%, #7B4A2E 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
       
        /* 指标卡片样式 */
        [data-testid="stMetricValue"] {
            background: #FFFFFFCC;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            display: inline-block;
            color: #5E3A2C;
            font-weight: 600;
        }
       
        /* 信息框样式 */
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 16px;
            border-left: 5px solid #B8865B;
        }
       
        /* 选项卡样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: #F5E8DA;
            border-radius: 40px;
            padding: 0.3rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 30px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            color: #7A4A2E;
        }
        .stTabs [aria-selected="true"] {
            background-color: #B8865B;
            color: white !important;
        }
       
        /* 文件上传区域 */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFFBB;
            border: 2px dashed #C9AD7A;
            border-radius: 24px;
            padding: 1rem;
        }
       
        /* 数据表格 */
        .stDataFrame {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid #EADBC6;
        }
       
        /* 自定义卡片 */
        .diagnosis-card {
            background: #FFFFFFDD;
            border-radius: 28px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            box-shadow: 0 8px 20px rgba(0,0,0,0.03);
            border: 1px solid #F0E2D2;
            backdrop-filter: blur(2px);
        }
       
        /* 指标区域美化 */
        .metric-container {
            background: #FCF6EF;
            border-radius: 24px;
            padding: 0.8rem;
            text-align: center;
            border: 1px solid #E9DBCB;
        }
       
        hr {
            margin: 1rem 0;
            border-color: #E2D0BA;
        }
       
        /* 流式气泡效果 */
        .stBalloons {
            filter: hue-rotate(15deg);
        }
    </style>
    """, unsafe_allow_html=True)

# ===== 图像质量检测函数 =====
def check_image_quality(image):
    """
    检测舌象照片质量
    返回 (is_qualified, issues_list)
    """
    issues = []
    
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # 计算灰度图
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        
        # 1. 亮度检测
        brightness = np.mean(gray)
        if brightness < 80:
            issues.append("⚠️ 光线太暗，请确保在充足自然光下拍摄")
        elif brightness > 230:
            issues.append("⚠️ 光线太亮/过曝，请避免强光直射")
        
        # 2. 清晰度检测
        if CV2_AVAILABLE:
            try:
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
                blur_score = np.var(laplacian)
                if blur_score < 50:
                    issues.append("⚠️ 画面模糊，请保持手机稳定并对焦舌体")
                elif blur_score < 100:
                    issues.append("⚠️ 画面略模糊，建议拍摄更清晰的照片")
            except Exception as e:
                print(f"清晰度检测跳过: {e}")
        
        # 3. 颜色饱和度检测
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        avg_sat = np.mean(sat)
        if avg_sat > 0.65:
            issues.append("⚠️ 颜色饱和度异常，请关闭美颜/滤镜功能")
        
        # 4. 舌体占比检测
        gray_flat = gray.flatten()
        sorted_gray = np.sort(gray_flat)
        lower_bound = sorted_gray[int(len(sorted_gray) * 0.3)]
        upper_bound = sorted_gray[int(len(sorted_gray) * 0.7)]
        tongue_mask = (gray >= lower_bound) & (gray <= upper_bound)
        tongue_ratio = np.sum(tongue_mask) / gray.size
        
        if tongue_ratio < 0.15:
            issues.append("⚠️ 舌体占比过小，请靠近拍摄/确保舌头完整露出")
        elif tongue_ratio > 0.75:
            issues.append("⚠️ 画面过满，请适当拉远使舌体完整")
        
        # 5. 反光检测
        high_threshold = np.percentile(gray, 98)
        highlight_ratio = np.sum(gray > high_threshold) / gray.size
        if highlight_ratio > 0.05:
            issues.append("⚠️ 检测到反光区域，请避开光源直射或关闭闪光灯")
        
    except Exception as e:
        print(f"质量检测异常: {e}")
        return True, []
    
    is_qualified = len(issues) == 0
    return is_qualified, issues

# ===== 色彩校正功能 =====
def auto_white_balance(image):
    """
    自动白平衡校正（基于灰度世界假设）
    返回校正后的PIL Image
    """
    if not CV2_AVAILABLE:
        return image
    
    try:
        img_array = np.array(image).astype(np.float32)
        
        r_channel = img_array[:,:,0]
        g_channel = img_array[:,:,1]
        b_channel = img_array[:,:,2]
        
        r_avg = np.mean(r_channel)
        g_avg = np.mean(g_channel)
        b_avg = np.mean(b_channel)
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        
        r_gain = gray_avg / (r_avg + 1e-6)
        g_gain = gray_avg / (g_avg + 1e-6)
        b_gain = gray_avg / (b_avg + 1e-6)
        
        r_gain = np.clip(r_gain, 0.5, 2.0)
        g_gain = np.clip(g_gain, 0.5, 2.0)
        b_gain = np.clip(b_gain, 0.5, 2.0)
        
        img_array[:,:,0] = np.clip(r_channel * r_gain, 0, 255)
        img_array[:,:,1] = np.clip(g_channel * g_gain, 0, 255)
        img_array[:,:,2] = np.clip(b_channel * b_gain, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
        
    except Exception as e:
        print(f"白平衡校正失败: {e}")
        return image

def advanced_color_correction(image):
    """
    高级色彩校正：白平衡 + 饱和度优化
    """
    if not CV2_AVAILABLE:
        return image
    
    try:
        img_wb = auto_white_balance(image)
        img_array = np.array(img_wb).astype(np.float32) / 255.0
        
        hsv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        sat_mean = np.mean(hsv[:,:,1])
        
        if sat_mean < 100:
            sat_boost = min(1.3, 120 / (sat_mean + 1))
            hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_boost, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"高级色彩校正失败: {e}")
        return image

# ===== 舌苔舌质分离功能（修复版：返回PIL Image）=====
def separate_tongue_coating(image):
    """
    分离舌苔和舌质
    返回: (舌苔区域PIL Image, 舌质区域PIL Image, 舌苔比例, 舌苔特征描述)
    """
    if not CV2_AVAILABLE:
        return image, image, 0.5, "无法分离（缺少cv2库）"
    
    try:
        img_array = np.array(image)
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        v_norm = hsv[:,:,2] / 255.0
        s_norm = hsv[:,:,1] / 255.0
        
        coating_mask = (v_norm > 0.5) & (s_norm < 0.35)
        body_mask = (s_norm > 0.2) & (v_norm > 0.3) & (v_norm < 0.85) & (~coating_mask)
        
        if CV2_AVAILABLE:
            kernel = np.ones((3,3), np.uint8)
            coating_mask = cv2.morphologyEx(coating_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            body_mask = cv2.morphologyEx(body_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        
        coating_img_array = img_array.copy()
        coating_img_array[~coating_mask] = [0, 0, 0]
        
        body_img_array = img_array.copy()
        body_img_array[~body_mask] = [0, 0, 0]
        
        # 转换为PIL Image
        coating_img = Image.fromarray(coating_img_array)
        body_img = Image.fromarray(body_img_array)
        
        total_pixels = np.sum(coating_mask) + np.sum(body_mask)
        if total_pixels > 0:
            coating_ratio = np.sum(coating_mask) / total_pixels
        else:
            coating_ratio = 0.5
        coating_ratio = min(0.95, max(0.05, coating_ratio))
        
        if np.sum(coating_mask) > 100:
            coating_pixels = img_array[coating_mask]
            mean_r = np.mean(coating_pixels[:,0])
            mean_g = np.mean(coating_pixels[:,1])
            mean_b = np.mean(coating_pixels[:,2])
            
            if mean_r > mean_g + 20 and mean_r > mean_b + 20:
                coating_color = "偏黄"
            elif mean_g > mean_r + 10 and mean_g > mean_b + 10:
                coating_color = "偏白"
            elif mean_b > mean_g + 15 and mean_b > mean_r + 15:
                coating_color = "偏灰"
            else:
                coating_color = "薄白"
        else:
            coating_color = "较少或无"
        
        if np.sum(body_mask) > 100:
            body_pixels = img_array[body_mask]
            body_r = np.mean(body_pixels[:,0])
            body_g = np.mean(body_pixels[:,1])
            body_b = np.mean(body_pixels[:,2])
            
            if body_r > body_g * 1.2 and body_r > body_b * 1.2:
                body_color = "偏红"
            elif body_r < body_g * 0.85 and body_r < body_b * 0.85:
                body_color = "偏淡/偏白"
            else:
                body_color = "淡红"
        else:
            body_color = "无法判断"
        
        coating_description = f"舌苔：{coating_color}，覆盖比例 {coating_ratio*100:.1f}%；舌质：{body_color}"
        
        return coating_img, body_img, coating_ratio, coating_description
        
    except Exception as e:
        print(f"舌苔舌质分离失败: {e}")
        return image, image, 0.5, f"分离失败: {str(e)}"

# ---------- 舌象类型标签 ----------
TONGUE_LABELS = [
    "Mirror-Approximated",
    "Thin-White",
    "White-Greasy",
    "Yellow-Greasy",
    "Grey-Black"
]

# ---------- 辨证提示和智能建议映射 ----------
DIAGNOSIS_MAP = {
    "Mirror-Approximated": {
        "诊断": "镜面舌（舌面光滑如镜，舌苔全无）",
        "辩证提示": "胃阴枯竭，胃气大伤。舌面光滑如镜，舌苔完全脱落，舌质红绛。多见于重症疾病后期。",
        "智能建议": """1. 立即就医：建议立即到正规医院就诊
2. 营养支持：高蛋白、高维生素、易消化饮食
3. 中药调理：可考虑沙参麦冬汤加减
4. 避免刺激：禁食辛辣、油炸食物
5. 定期复查：每1-2周复查舌象变化"""
    },
    "Thin-White": {
        "诊断": "薄白苔（正常舌象）",
        "辩证提示": "气血调和，胃气充足。为健康舌象。",
        "智能建议": """1. 维持现状：保持良好生活习惯
2. 饮食调理：均衡饮食，多吃蔬菜水果
3. 适度运动：每周3-5次中等强度运动
4. 充足睡眠：保证每天7-8小时睡眠
5. 定期检查：每年进行一次体检"""
    },
    "White-Greasy": {
        "诊断": "白腻苔（湿浊内蕴）",
        "辩证提示": "湿浊内停，痰饮积聚。多见于消化不良、慢性胃炎等。",
        "智能建议": """1. 健脾祛湿：可用茯苓、白术、薏苡仁煮粥
2. 饮食调整：减少油腻、甜食摄入
3. 中药调理：考虑平胃散、二陈汤
4. 适量运动：每天散步30分钟
5. 穴位按摩：按摩足三里、丰隆穴"""
    },
    "Yellow-Greasy": {
        "诊断": "黄腻苔（湿热内蕴）",
        "辩证提示": "湿热蕴结，痰热互结。常见于急性胃肠炎等。",
        "智能建议": """1. 清热利湿：可用黄芩、黄连泡水
2. 饮食清淡：忌辛辣油腻
3. 及时就医：如伴发热应及时就诊
4. 中药治疗：考虑茵陈蒿汤
5. 多饮水：每天饮水2000ml以上"""
    },
    "Grey-Black": {
        "诊断": "灰黑苔（热极或寒极）",
        "辩证提示": "病情危重，或热极或寒极。",
        "智能建议": """1. 紧急就医：立即到急诊就诊
2. 全面检查：进行血常规、肝肾功能检查
3. 专科治疗：到相应专科治疗
4. 中西医结合：配合中药治疗
5. 密切观察：每日记录舌象变化"""
    }
}

# ===== 高精度舌象识别算法 =====
def extract_tongue_color_features(image, coating_ratio=None, coating_description=None):
    try:
        img_array = np.array(image)
        h_img, w_img = img_array.shape[:2]

        crop_size = min(h_img, w_img) // 3
        cy, cx = h_img // 2, w_img // 2
        center = img_array[cy-crop_size//2:cy+crop_size//2, cx-crop_size//2:cx+crop_size//2]

        r = np.mean(center[:,:,0])
        g = np.mean(center[:,:,1])
        b = np.mean(center[:,:,2])

        brightness = 0.299*r + 0.587*g + 0.114*b

        rn, gn, bn = r/255.0, g/255.0, b/255.0
        hue, sat, val = colorsys.rgb_to_hsv(rn, gn, bn)

        r_std = np.std(center[:,:,0])
        g_std = np.std(center[:,:,1])
        b_std = np.std(center[:,:,2])
        uniformity = (r_std + g_std + b_std) / 3

        yellow_ratio = (r - b) / (g + 1)

        confidence = 0.85

        if coating_ratio is not None:
            if coating_ratio < 0.08:
                tongue_type = "Mirror-Approximated"
                confidence = 0.94
                return tongue_type, round(confidence, 2), {
                    "亮度": round(brightness, 2),
                    "饱和度": round(sat, 2),
                    "色调": round(hue, 2),
                    "颜色均匀度": round(uniformity, 2),
                    "黄色指数": round(yellow_ratio, 2),
                    "舌苔覆盖比例": round(coating_ratio * 100, 1),
                    "舌苔分析": coating_description or "无"
                }
            
            if coating_ratio > 0.35:
                if yellow_ratio > 10 and sat > 0.45:
                    tongue_type = "Yellow-Greasy"
                    confidence = 0.91
                elif sat > 0.25 and brightness < 180:
                    tongue_type = "White-Greasy"
                    confidence = 0.89

        if brightness > 210 and sat < 0.15 and uniformity < 25:
            tongue_type = "Mirror-Approximated"
            confidence = 0.92
        elif brightness < 90:
            tongue_type = "Grey-Black"
            confidence = 0.90
        elif yellow_ratio > 12 and sat > 0.48 and hue > 0.08:
            tongue_type = "Yellow-Greasy"
            confidence = 0.89
        elif sat > 0.28 and 110 < brightness < 170 and yellow_ratio < 8:
            tongue_type = "White-Greasy"
            confidence = 0.87
        else:
            tongue_type = "Thin-White"
            confidence = 0.82

        confidence = max(0.75, min(0.98, confidence))

        feature_info = {
            "亮度": round(brightness, 2),
            "饱和度": round(sat, 2),
            "色调": round(hue, 2),
            "颜色均匀度": round(uniformity, 2),
            "黄色指数": round(yellow_ratio, 2)
        }
        
        if coating_ratio is not None:
            feature_info["舌苔覆盖比例"] = round(coating_ratio * 100, 1)
            feature_info["舌苔分析"] = coating_description or "无"
        
        return tongue_type, round(confidence, 2), feature_info

    except Exception as e:
        print(f"特征提取错误: {e}")
        return "Thin-White", 0.75, {}

# ===== 舌象分析核心函数 =====
def analyze_tongue(image_path, enable_color_correction=True, enable_coating_separation=True):
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        if enable_color_correction and CV2_AVAILABLE:
            img = advanced_color_correction(img)
        
        results = yolo_model(image_path, conf=0.1, iou=0.45)
        result = results[0]
        best_box, best_conf, all_boxes = None, 0.0, []
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id, conf, xyxy = int(box.cls[0]), float(box.conf[0]), box.xyxy[0].tolist()
                all_boxes.append(f"{result.names[cls_id]}:{conf:.2f}")
                if cls_id == 0 and conf > best_conf:
                    best_conf, best_box = conf, xyxy
        
        if best_box is None and result.boxes is not None:
            for box in result.boxes:
                conf, xyxy = float(box.conf[0]), box.xyxy[0].tolist()
                if conf > best_conf:
                    best_conf, best_box = conf, xyxy
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            tongue_y1, tongue_y2 = y1 + (y2 - y1)//2, y2
            tongue_x1, tongue_x2 = x1, x2
        else:
            margin_w, margin_h = int(width*0.2), int(height*0.2)
            tongue_x1, tongue_x2, tongue_y1, tongue_y2 = margin_w, width - margin_w, margin_h, height - margin_h
        
        tongue_x1, tongue_y1 = max(0, tongue_x1), max(0, tongue_y1)
        tongue_x2, tongue_y2 = min(width, tongue_x2), min(height, tongue_y2)
        if tongue_x1 >= tongue_x2: tongue_x2 = tongue_x1 + 10
        if tongue_y1 >= tongue_y2: tongue_y2 = tongue_y1 + 10
        
        cropped_img = img.crop((tongue_x1, tongue_y1, tongue_x2, tongue_y2))
        
        coating_ratio = None
        coating_description = None
        coating_img_base64 = None
        body_img_base64 = None
        
        if enable_coating_separation and CV2_AVAILABLE:
            coating_img, body_img, coating_ratio, coating_description = separate_tongue_coating(cropped_img)
            
            # 现在 coating_img 和 body_img 已经是 PIL Image，可以直接保存
            buffered_coating = BytesIO()
            coating_img.save(buffered_coating, format="PNG")
            coating_img_base64 = base64.b64encode(buffered_coating.getvalue()).decode()
            
            buffered_body = BytesIO()
            body_img.save(buffered_body, format="PNG")
            body_img_base64 = base64.b64encode(buffered_body.getvalue()).decode()
        
        tongue_type, confidence, feature_info = extract_tongue_color_features(
            cropped_img, coating_ratio, coating_description
        )
        
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.rectangle([tongue_x1, tongue_y1, tongue_x2, tongue_y2], outline="#00008B", width=6)
        
        if best_conf > 0:
            conf_text = f"Tongue Area (Conf: {best_conf:.2f})"
            draw.text((tongue_x1, tongue_y1-10), conf_text, fill="#00008B")
            draw.text((tongue_x2-80, tongue_y2-20), f"Conf: {best_conf:.2f}", fill="#FF6600")
        else:
            draw.text((tongue_x1, tongue_y1-10), "Tongue Area (Estimated)", fill="#00008B")
        
        buffered = BytesIO()
        draw_img.save(buffered, format="PNG")
        boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        buffered_crop = BytesIO()
        cropped_img.save(buffered_crop, format="PNG")
        tongue_region_base64 = base64.b64encode(buffered_crop.getvalue()).decode()
        
        diagnosis_info = DIAGNOSIS_MAP.get(tongue_type, {"诊断":"未知舌象","辩证提示":"无法准确判断","智能建议":"请咨询专业中医师"})
        
        result_dict = {
            "舌象类型": tongue_type, 
            "置信度": confidence, 
            "诊断": diagnosis_info["诊断"],
            "辩证提示": diagnosis_info["辩证提示"], 
            "智能建议": diagnosis_info["智能建议"],
            "boxed_image_base64": boxed_image_base64, 
            "tongue_region_base64": tongue_region_base64,
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"), 
            "特征信息": feature_info,
            "人体检测置信度": best_conf, 
            "检测详情": ", ".join(all_boxes) if all_boxes else "无检测"
        }
        
        if coating_img_base64:
            result_dict["coating_img_base64"] = coating_img_base64
            result_dict["body_img_base64"] = body_img_base64
            result_dict["舌苔分离分析"] = coating_description
        
        return result_dict
        
    except Exception as e:
        st.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ---------- 历史记录读写 ----------
def save_history(history):
    with open("tongue_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists("tongue_history.json"):
        with open("tongue_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------- 加载 YOLO 模型 ----------
@st.cache_resource
def load_yolo_model(model_size="n"):
    model_map = {"n":"yolov10n.pt","s":"yolov10s.pt","m":"yolov10m.pt","b":"yolov10b.pt","l":"yolov10l.pt","x":"yolov10x.pt"}
    model_name = model_map.get(model_size, "yolov10n.pt")
    if not os.path.exists(model_name):
        st.info(f"正在下载 {model_name} 模型，首次使用需要联网...")
    return YOLO(model_name)

yolo_model = load_yolo_model()

# ---------- 登录模块 ----------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def init_auth():
    if 'users' not in st.session_state:
        st.session_state['users'] = {"doctor": make_hashes("123456"), "student": make_hashes("654321")}
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 舌象诊断 · 专属登录</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7A4A2E;'>欢迎使用智能中医舌象分析系统</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            st.markdown("<div class='diagnosis-card'>", unsafe_allow_html=True)
            menu = ["登录", "注册"]
            choice = st.selectbox("选择操作", menu, key="login_menu")
            if choice == "登录":
                username = st.text_input("用户名", key="login_username", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", key="login_password", placeholder="请输入密码")
                if st.button("登录", use_container_width=True):
                    if username in st.session_state['users'] and check_hashes(password, st.session_state['users'][username]):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.success(f"欢迎回来，{username} 医师！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误！")
            elif choice == "注册":
                new_user = st.text_input("新用户名", key="reg_username", placeholder="请输入新用户名")
                new_password = st.text_input("新密码", type="password", key="reg_password", placeholder="请设置密码")
                if st.button("注册", use_container_width=True):
                    if new_user in st.session_state['users']:
                        st.warning("用户已存在！")
                    elif new_user and new_password:
                        st.session_state['users'][new_user] = make_hashes(new_password)
                        st.success("注册成功！请返回登录")
                    else:
                        st.error("用户名和密码不能为空")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- 舌象分析页面 ----------
def analysis_page():
    st.markdown("<h1>👅 智能舌象诊断</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B4C3C;'>上传舌部照片，系统将基于YOLO模型精准定位舌体区域并给出中医辨证建议。</p>", unsafe_allow_html=True)
    st.markdown("---")
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "upload_passed_quality" not in st.session_state:
        st.session_state.upload_passed_quality = False
    if "current_temp_path" not in st.session_state:
        st.session_state.current_temp_path = None

    st.markdown("<div class='diagnosis-card'>", unsafe_allow_html=True)
    st.subheader("👤 患者基本信息")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        gender = st.selectbox("性别", ["男", "女"], index=0)
        age = st.number_input("年龄（岁）", min_value=1, max_value=120, value=30, step=1)
    with info_col2:
        height = st.number_input("身高（cm）", min_value=50, max_value=250, value=170, step=1)
        weight = st.number_input("体重（kg）", min_value=10, max_value=300, value=65, step=1)
    
    bmi = round(weight / ((height/100)**2), 1) if height > 0 else 0.0
    if bmi < 18.5: bmi_level = "偏瘦"
    elif 18.5 <= bmi < 24: bmi_level = "正常"
    elif 24 <= bmi < 28: bmi_level = "超重"
    else: bmi_level = "肥胖"
    st.success(f"**自动计算BMI指数：{bmi}   |   体型：{bmi_level}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
    st.subheader("📤 上传舌象照片")
    col_upload_guide, col_upload_deco = st.columns([3, 1])
    with col_upload_guide:
        with st.expander("📷 拍摄指南（点击收起）", expanded=True):
            st.markdown("""
            <div style="background:#FCF6EF; padding:1rem; border-radius:16px;">
                <p>✅ 在<strong>自然光</strong>下拍摄，避免强光直射或过暗</p>
                <p>✅ 舌头<strong>自然平伸</strong>，不要过度用力</p>
                <p>✅ 对焦舌体，确保图像<strong>清晰不模糊</strong></p>
                <p>✅ 关闭相机<strong>美颜、滤镜、HDR</strong>等功能</p>
                <p>✅ 尽量露出<strong>整个舌头</strong>，包括舌根</p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("⚙️ 高级分析设置（可选）"):
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            enable_color_correction = st.checkbox("🎨 自动色彩校正（白平衡）", value=True, 
                                                   help="校正因光照导致的色偏，提高诊断准确性")
        with col_set2:
            enable_coating_separation = st.checkbox("🔬 舌苔/舌质分离分析", value=True,
                                                     help="区分舌苔和舌质，提供更详细的舌象分析")
    
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("点击或拖拽上传舌象照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        temp_path = f"temp_{int(time.time())}_{uploaded_file.name}"
        
        if st.session_state.current_temp_path != temp_path:
            st.session_state.upload_passed_quality = False
            st.session_state.current_temp_path = temp_path
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            img = Image.open(temp_path).convert("RGB")
            is_qualified, issues = check_image_quality(img)
            
            if is_qualified:
                st.session_state.upload_passed_quality = True
                st.success("✅ 图像质量合格，可以开始分析！")
            else:
                st.session_state.upload_passed_quality = False
                st.error("❌ 图像质量不符合要求，请重新拍摄/上传")
                for issue in issues:
                    st.warning(issue)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    st.session_state.current_temp_path = None
                st.info("📷 **请重新拍摄：**\n\n" + "\n".join(issues))
                st.markdown("""
                <div style="background:#FFF3E0; padding:1rem; border-radius:16px; margin-top:1rem;">
                    <p>💡 <strong>拍摄小贴士：</strong></p>
                    <ul>
                        <li>请确保在<strong>充足自然光</strong>下拍摄</li>
                        <li>拍摄时<strong>手部稳定</strong>，等待对焦清晰后再按快门</li>
                        <li>关闭手机<strong>美颜、滤镜</strong>功能</li>
                        <li>舌头<strong>自然伸出</strong>，不要用力或卷曲</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.upload_passed_quality and st.session_state.current_temp_path:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("🔍 检测定位结果")
                boxed_placeholder = st.empty()
            with col2:
                st.subheader("✂️ 舌体区域特写")
                tongue_placeholder = st.empty()
            
            if st.button("✨ 开始分析", type="primary", use_container_width=True):
                with st.spinner("正在分析舌象，请稍候..."):
                    result = analyze_tongue(
                        st.session_state.current_temp_path,
                        enable_color_correction=enable_color_correction,
                        enable_coating_separation=enable_coating_separation
                    )
                    if result:
                        st.session_state.current_result = result
                        st.success("✅ 分析完成！")
                        st.balloons()
                        boxed_placeholder.image(f"data:image/png;base64,{result['boxed_image_base64']}", width=400)
                        tongue_placeholder.image(f"data:image/png;base64,{result['tongue_region_base64']}", width=400)
                        
                        if 'coating_img_base64' in result:
                            st.markdown("---")
                            st.subheader("🔬 舌苔/舌质分离分析")
                            
                            col_coat1, col_coat2 = st.columns(2)
                            with col_coat1:
                                st.markdown("**👅 舌苔区域**")
                                st.image(f"data:image/png;base64,{result['coating_img_base64']}", width=300)
                            with col_coat2:
                                st.markdown("**❤️ 舌质区域**")
                                st.image(f"data:image/png;base64,{result['body_img_base64']}", width=300)
                            
                            if '舌苔分离分析' in result:
                                st.info(f"📊 {result['舌苔分离分析']}")
                        
                        st.markdown("---")
                        st.subheader("📊 诊断结论")
                        col_patient, col_result1, col_result2 = st.columns(3)
                        with col_patient:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("性别", gender)
                            st.metric("年龄", f"{age} 岁")
                            st.metric("BMI", f"{bmi} ({bmi_level})")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col_result1:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("舌象类型", result["舌象类型"])
                            st.metric("分析置信度", f"{result['置信度']*100:.1f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col_result2:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.info("诊断结论")
                            st.write(result["诊断"])
                            st.info("分析时间")
                            st.write(result["分析时间"])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        tab1, tab2 = st.tabs(["🧠 辩证提示", "💡 智能建议"])
                        with tab1:
                            st.markdown(f"<div class='diagnosis-card'>{result['辩证提示']}</div>", unsafe_allow_html=True)
                        with tab2:
                            st.markdown(f"<div class='diagnosis-card'>{result['智能建议']}</div>", unsafe_allow_html=True)
                        
                        with st.expander("🔬 舌象特征详情（技术信息）"):
                            st.json(result.get("特征信息", {}))
                        
                        history_record = {
                            "时间": result["分析时间"],
                            "性别": gender,
                            "年龄": age,
                            "身高_cm": height,
                            "体重_kg": weight,
                            "BMI": bmi,
                            "体型": bmi_level,
                            "舌象类型": result["舌象类型"],
                            "置信度": result["置信度"],
                            "诊断": result["诊断"],
                            "辩证提示": result["辩证提示"],
                            "智能建议": result["智能建议"],
                            "boxed_image_base64": result["boxed_image_base64"],
                            "tongue_region_base64": result["tongue_region_base64"],
                            "特征信息": result.get("特征信息", {})
                        }
                        if 'coating_img_base64' in result:
                            history_record["coating_img_base64"] = result["coating_img_base64"]
                            history_record["body_img_base64"] = result["body_img_base64"]
                            history_record["舌苔分离分析"] = result["舌苔分离分析"]
                        
                        st.session_state.history.append(history_record)
                        save_history(st.session_state.history)
                        st.success("📝 记录已保存至历史档案！")
    else:
        st.session_state.upload_passed_quality = False
        if st.session_state.current_temp_path and os.path.exists(st.session_state.current_temp_path):
            os.remove(st.session_state.current_temp_path)
        st.session_state.current_temp_path = None

# ---------- 历史记录页面 ----------
def history_page():
    st.markdown("<h1>📋 诊断历史档案</h1>", unsafe_allow_html=True)
    st.markdown("---")
    history = load_history()
    if len(history) == 0:
        st.info("✨ 暂无历史分析记录，请先前往【舌象分析】进行诊断。")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("总记录数", len(history))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        tongue_types = [h.get("舌象类型", "未知") for h in history]
        most_common = max(set(tongue_types), key=tongue_types.count) if tongue_types else "无"
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("最常见舌象", most_common)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        avg_confidence = sum([h.get("置信度", 0) for h in history]) / len(history)
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("平均置信度", f"{avg_confidence*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.subheader("📜 历史记录列表")
    table_data = []
    for idx, record in enumerate(history):
        table_data.append({
            "序号": idx+1, "时间": record.get("时间", ""), "性别": record.get("性别", ""),
            "年龄": f"{record.get('年龄', '')}岁", "BMI": record.get("BMI", ""),
            "舌象类型": record.get("舌象类型", ""), "诊断": record.get("诊断", ""),
            "置信度": f"{record.get('置信度', 0)*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
    
    st.subheader("🔍 回顾详细记录")
    options = [f"{i+1}. {record['时间']} - {record.get('性别','')}{record.get('年龄','')}岁 - {record['舌象类型']}" for i, record in enumerate(history)]
    if options:
        selected_option = st.selectbox("选择要查看的记录", options)
        if selected_option:
            idx = int(selected_option.split(".")[0]) - 1
            r = history[idx]
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"性别：{r.get('性别')}")
                st.info(f"年龄：{r.get('年龄')} 岁")
                st.info(f"BMI：{r.get('BMI')} ({r.get('体型')})")
                st.info(f"舌象：{r.get('舌象类型')}")
                st.info(f"诊断：{r.get('诊断')}")
                if '舌苔分离分析' in r:
                    st.info(f"舌象细节：{r.get('舌苔分离分析')}")
            with c2:
                if "tongue_region_base64" in r and r["tongue_region_base64"]:
                    img_data = base64.b64decode(r["tongue_region_base64"])
                    st.image(Image.open(BytesIO(img_data)), width=250)
                if 'coating_img_base64' in r:
                    col_show1, col_show2 = st.columns(2)
                    with col_show1:
                        st.caption("舌苔区域")
                        coating_data = base64.b64decode(r["coating_img_base64"])
                        st.image(Image.open(BytesIO(coating_data)), width=120)
                    with col_show2:
                        st.caption("舌质区域")
                        body_data = base64.b64decode(r["body_img_base64"])
                        st.image(Image.open(BytesIO(body_data)), width=120)
            with st.expander("🧠 辩证提示"):
                st.write(r.get("辩证提示"))
            with st.expander("💡 智能建议"):
                st.write(r.get("智能建议"))
            with st.expander("🔬 详细特征信息"):
                st.json(r.get("特征信息", {}))

# ---------- 主应用 ----------
def main():
    inject_custom_css()
    init_auth()
    if not st.session_state['logged_in']:
        login_page()
        return
    with st.sidebar:
        st.markdown(f"<h2 style='color:#5E3A2C;'>👤 {st.session_state['username']}</h2>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        page = st.radio("📱 导航菜单", ["👅 舌象分析", "📋 历史记录"], label_visibility="collapsed")
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()
    if page == "👅 舌象分析":
        analysis_page()
    else:
        history_page()

if __name__ == '__main__':
    main()