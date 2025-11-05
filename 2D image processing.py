import os
import numpy as np
import nibabel as nib
import cv2

# 定义目录路径
image_dir = 'D:/xiaojuan/V2/images'
mask_dir = 'D:/xiaojuan/V2/masks'
save_root_dir = 'D:/xiaojuan/V2/2D_custom'  # 总输出目录

# ============ 自定义参数：提取最大ROI上下几层 ===========
extract_range = [-2, -1, 0, 1, 2]  # 提取最大ROI层 ±2 层（共5个切片）
# =====================================================

# ============ CT窗宽窗位设置（针对腹部增强CT）============
WINDOW_CENTER = 60  # 窗位：适合腹部软组织和囊实性病灶
WINDOW_WIDTH = 350  # 窗宽：可显示囊性（低密度）到实性（高密度）成分
# 可选预设：
# 软组织窗：WC=40-60, WW=300-400
# 肝脏窗：WC=60, WW=150
# 如需更亮：增大WINDOW_CENTER；如需更高对比度：减小WINDOW_WIDTH
BRIGHTNESS_ADJUST = 1.0  # 整体亮度调整系数（1.0=不变，>1增亮，<1变暗）
CONTRAST_ADJUST = 1.0    # 对比度调整系数（1.0=不变，>1增强对比度）
# =====================================================

def apply_ct_window(image_slice, window_center, window_width):
    """
    应用CT窗宽窗位
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    windowed = np.clip(image_slice, min_value, max_value)
    windowed = (windowed - min_value) / (max_value - min_value) * 255
    return windowed.astype(np.uint8)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """
    调整亮度和对比度
    brightness: 亮度系数 (>1 增亮, <1 变暗)
    contrast: 对比度系数 (>1 增强对比度, <1 降低对比度)
    """
    # 转换为float进行计算
    image = image.astype(np.float32)
    
    # 调整对比度（围绕中心值）
    image = (image - 127.5) * contrast + 127.5
    
    # 调整亮度
    image = image * brightness
    
    # 限制范围并转换回uint8
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# 获取图像文件列表
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

# 创建主保存目录
os.makedirs(save_root_dir, exist_ok=True)

for rel_index in extract_range:
    save_dir = os.path.join(save_root_dir, f"2D_{'0' if rel_index==0 else (('+' if rel_index>0 else '') + str(rel_index))}")
    os.makedirs(save_dir, exist_ok=True)

for image_file in image_files:
    # 路径
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file)
    
    # 加载
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image_data = image_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    assert image_data.shape == mask_data.shape, f"Shape Mismatch: {image_file}"
    
    num_slices = mask_data.shape[2]
    roi_areas = [np.sum(mask_data[:, :, z]) for z in range(num_slices)]
    max_area_slice = np.argmax(roi_areas)
    
    # 计算最大ROI的边界框
    max_slice_mask = mask_data[:, :, max_area_slice]
    max_rows, max_cols = np.where(max_slice_mask > 0)
    if len(max_rows) == 0 or len(max_cols) == 0:
        print(f"错误: 最大ROI层 {max_area_slice} 在 {image_file} 中没有ROI")
        continue
    
    expand = 3
    max_min_row, max_max_row = max(0, np.min(max_rows) - expand), min(max_slice_mask.shape[0] - 1, np.max(max_rows) + expand)
    max_min_col, max_max_col = max(0, np.min(max_cols) - expand), min(max_slice_mask.shape[1] - 1, np.max(max_cols) + expand)
    
    # 遍历自定义的提取层
    for rel_index in extract_range:
        z = max_area_slice + rel_index
        if z < 0 or z >= num_slices:
            print(f"跳过 {image_file} 的相对索引 {rel_index}（实际索引 {z}）: 越界")
            continue

        mask_slice = mask_data[:, :, z]
        image_slice = image_data[:, :, z]
        rows, cols = np.where(mask_slice > 0)
        if len(rows) == 0 or len(cols) == 0:
            print(f"警告：{image_file} 的 2D_{rel_index} 层没有ROI，使用最大层ROI边界")
            min_row, max_row = max_min_row, max_max_row
            min_col, max_col = max_min_col, max_max_col
        else:
            min_row, max_row = max(0, np.min(rows) - expand), min(mask_slice.shape[0] - 1, np.max(rows) + expand)
            min_col, max_col = max(0, np.min(cols) - expand), min(mask_slice.shape[1] - 1, np.max(cols) + expand)

        cropped_image = image_slice[min_row:max_row+1, min_col:max_col+1]
        
        # ========== 针对腹部CT增强囊实性病灶的图像处理 ==========
        # 1. 应用CT窗宽窗位
        cropped_image = apply_ct_window(cropped_image, WINDOW_CENTER, WINDOW_WIDTH)
        
        # 2. 调整亮度和对比度
        cropped_image = adjust_brightness_contrast(cropped_image, 
                                                   brightness=BRIGHTNESS_ADJUST, 
                                                   contrast=CONTRAST_ADJUST)
        
        # 3. 轻度锐化增强病灶边界（可选）
        blurred = cv2.GaussianBlur(cropped_image, (3, 3), 0)
        cropped_image = cv2.addWeighted(cropped_image, 1.3, blurred, -0.3, 0)
        
        # 4. 可选：CLAHE自适应直方图均衡（增强局部对比度）
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # cropped_image = clahe.apply(cropped_image)
        
        # 确保数据类型正确
        cropped_image = np.clip(cropped_image, 0, 255).astype(np.uint8)
        # =======================================================
        
        # 旋转90度
        cropped_image = np.rot90(cropped_image, k=1)

        save_dir = os.path.join(save_root_dir, f"2D_{'0' if rel_index==0 else (('+' if rel_index>0 else '') + str(rel_index))}")
        base_name = os.path.splitext(image_file)[0]
        save_name = f"{base_name}.png"
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, cropped_image)
        print(f"已保存: {save_path}")

print("处理完成，自定义层数的2D图像已保存至指定目录。")
