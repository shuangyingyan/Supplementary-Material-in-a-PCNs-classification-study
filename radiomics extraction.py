
import os
import sys
import warnings
import numpy as np
import pandas as pd
import logging
import SimpleITK as sitk
from radiomics import featureextractor
import copy
from contextlib import redirect_stderr
from tqdm import tqdm

# ---- 关闭非关键警告/日志 ----
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('radiomics').setLevel(logging.CRITICAL)  # 仅显示严重错误
logging.getLogger('SimpleITK').setLevel(logging.CRITICAL)

# ================== 路径与基本配置 ==================
# 输入目录
image_dir = r"C:\new1\ICC A\images"
mask_dir = r"C:\new1\ICC A\Aafter"
modality = 'CT'  # 影像序列名称

# 输出目录和文件
output_dir = r"D:\xiaojuan\A2\CT_radiomics"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'radiomics_features_all_casesICCA后.csv')

# 目标标签值（最终会将掩膜统一为这个标签）
TARGET_LABEL = 1

# ================== 掩膜清洗/重映射策略 ==================
# 候选标签：如果掩膜中不含1，则按此列表优先级寻找（常见255）
ALT_LABEL_CANDIDATES = [1, 255, 2, 3]
# 若仍未找到，则选择“非零标签中体素最多”的标签作为ROI
FALLBACK_USE_LARGEST_LABEL = True
# 如果掩膜是浮点型（如概率图），首先按阈值二值化；若为None则直接用“非零即ROI”
FLOAT_MASK_THRESHOLD = 0.5
# 二值化后是否只保留最大连通域（多发病灶建议设置为 False；若噪点多可设置 True）
KEEP_ONLY_LARGEST_COMPONENT = False

# 体素数阈值
MIN_VOXELS_FOR_FEATURES = 1  # 少于该体素数的ROI不做特征提取
MIN_VOXELS_FOR_ROI_SAVE = 1   # 保留此变量但不再使用

# 取消裁剪（仅保存全尺寸ROI）
SAVE_CROPPED_ROI = False  # 已无效，因为不保存任何ROI
ROI_CROP_MARGIN_VOX = 0   # 已无效

# 调试输出：打印每个病例掩膜标签统计
DEBUG_PRINT_MASK_LABELS = False

# 取消重采样：若掩膜与影像几何不匹配，跳过该病例
ENABLE_MASK_RESAMPLING = False  # 已取消重采样

# ================== Radiomics 参数 (适配CT) ==================
BASE_RADIOMICS_PARAMS = {
    'imageType': {
        'Original': {},
        'Wavelet': {},
        'LoG': {'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]},
        'Square': {},
        'SquareRoot': {},
        'Logarithm': {},
        'Exponential': {},
        'Gradient': {}
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
        'glrlm': [],
        'glszm': [],
        'gldm': [],
        'ngtdm': [],
        'shape': []
    },
    'setting': {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'normalize': False,
        'normalizeScale': 100,
        'correctMask': True,
        'additionalInfo': True,
        'force2D': False
    }
}

# ================== 工具函数 ==================

INTEGER_PIXEL_IDS = {
    sitk.sitkUInt8, sitk.sitkInt8,
    sitk.sitkUInt16, sitk.sitkInt16,
    sitk.sitkUInt32, sitk.sitkInt32,
    sitk.sitkUInt64, sitk.sitkInt64
}

def is_integer_image(itk_img: sitk.Image) -> bool:
    return itk_img.GetPixelID() in INTEGER_PIXEL_IDS

def _to_scalar(v):
    """将Numpy标量或数组转换为Python原生float类型，便于存储。"""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v

def print_mask_summary(case_id: str, mask_itk: sitk.Image, prefix: str = "INFO"):
    """打印掩膜中的标签分布和体素数以便调试。"""
    try:
        if is_integer_image(mask_itk):
            lss = sitk.LabelShapeStatisticsImageFilter()
            lss.Execute(mask_itk)
            labels = list(lss.GetLabels())
            labels_sorted = sorted(int(l) for l in labels)
            msg = [f"[{prefix}] 病例 {case_id} 掩膜标签统计: 总标签数={len(labels_sorted)}"]
            for lab in labels_sorted:
                msg.append(f"  - Label={int(lab)} Voxels={int(lss.GetNumberOfPixels(lab))}")
            print("\n".join(msg))
        else:
            arr = sitk.GetArrayViewFromImage(mask_itk)
            nonzero = int(np.count_nonzero(arr))
            uniq = np.unique(np.asarray(arr))
            print(f"[{prefix}] 病例 {case_id} 掩膜为非整数类型: 非零体素={nonzero}, 唯一值样本(前10)={uniq[:10]}")
    except Exception as e:
        print(f"[{prefix}] 病例 {case_id} 掩膜统计失败: {e}")

def make_extractor(params: dict) -> featureextractor.RadiomicsFeatureExtractor:
    """根据参数字典创建并返回一个特征提取器实例。"""
    try:
        ext = featureextractor.RadiomicsFeatureExtractor(params)
        return ext
    except Exception as e:
        print(f"[ERROR] 创建提取器失败: {e}")
        return None

extractor_base = make_extractor(BASE_RADIOMICS_PARAMS)

def _filter_log_sigmas_for_image(image_itk: sitk.Image, sigmas: list) -> list:
    """
    过滤掉对于当前图像尺寸而言过大的LoG sigma值，以避免PyRadiomics报错。
    """
    size = np.array(image_itk.GetSize(), dtype=np.int64)
    spacing = np.array(image_itk.GetSpacing(), dtype=np.float64)
    safe_sigmas = []
    for s in sigmas:
        s = float(s)
        # LoG核大小粗估：6 * (sigma / spacing) + 1
        kernel_dim_estimate = (6.0 * (s / spacing)) + 1.0
        if np.all(kernel_dim_estimate <= size.astype(np.float64)):
            safe_sigmas.append(s)
    return safe_sigmas

def _build_case_extractor(image_itk: sitk.Image) -> featureextractor.RadiomicsFeatureExtractor:
    """
    基于当前图像的几何信息，动态构建一个合适的特征提取器（适配LoG sigma）。
    """
    if not extractor_base:
        return None
    
    p = copy.deepcopy(BASE_RADIOMICS_PARAMS)
    if 'LoG' in p.get('imageType', {}) and 'sigma' in p['imageType']['LoG']:
        sigmas = p['imageType']['LoG']['sigma']
        safe_sigmas = _filter_log_sigmas_for_image(image_itk, sigmas)
        
        if not safe_sigmas:
            del p['imageType']['LoG']
        elif len(safe_sigmas) < len(sigmas):
            p['imageType']['LoG']['sigma'] = safe_sigmas
        
        # 只有当参数实际被修改时，才需要重新创建提取器
        if p == BASE_RADIOMICS_PARAMS:
            return extractor_base
        else:
            return make_extractor(p)
    return extractor_base

def read_image_and_mask(image_path: str, mask_path: str):
    """
    读取影像与掩膜。影像以Float32加载，掩膜保留原类型（稍后统一清洗）。
    """
    img = sitk.ReadImage(image_path, sitk.sitkFloat32)
    msk = sitk.ReadImage(mask_path)
    return img, msk

def check_geometry_match(mask_itk: sitk.Image, image_itk: sitk.Image) -> bool:
    """
    检查几何（尺寸、间距、原点、方向）是否匹配。
    """
    return (mask_itk.GetSize() == image_itk.GetSize() and
            mask_itk.GetSpacing() == image_itk.GetSpacing() and
            mask_itk.GetOrigin() == image_itk.GetOrigin() and
            mask_itk.GetDirection() == image_itk.GetDirection())

def binarize_float_mask(mask_itk: sitk.Image, threshold: float, case_id: str):
    """
    对浮点掩膜进行二值化。若按阈值后为空，则退化为“非零即ROI”。
    """
    # 阈值化
    if threshold is not None:
        mask_bin = sitk.Cast(sitk.GreaterEqual(mask_itk, float(threshold)), sitk.sitkUInt8)
        vox = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_bin)))
        if vox > 0:
            return mask_bin, vox, f"float>=thr({threshold})"
    # 退化为非零
    mask_bin = sitk.Cast(sitk.NotEqual(mask_itk, 0.0), sitk.sitkUInt8)
    vox = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_bin)))
    return mask_bin, vox, "float_nonzero"

def keep_largest_component(mask_bin: sitk.Image) -> sitk.Image:
    """
    保留二值掩膜的最大连通域。
    """
    cc = sitk.ConnectedComponent(mask_bin)
    rcc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(rcc, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    return sitk.Cast(largest, sitk.sitkUInt8)

def sanitize_and_unify_mask(mask_itk: sitk.Image,
                            desired_label: int,
                            alt_candidates: list,
                            fallback_largest: bool,
                            float_threshold: float,
                            keep_largest: bool,
                            case_id: str):
    """
    将掩膜统一清洗为二值(Label=1)，返回 (mask_bin_uint8, info_dict)
    info_dict 包含：strategy, used_label, voxel_count, found_labels
    """
    info = {'strategy': None, 'used_label': None, 'voxel_count': 0, 'found_labels': None}

    try:
        if is_integer_image(mask_itk):
            # 用 LabelShapeStatistics 统计
            lss = sitk.LabelShapeStatisticsImageFilter()
            lss.Execute(mask_itk)
            labels = sorted(int(l) for l in lss.GetLabels())
            info['found_labels'] = labels

            # 去掉背景0
            nz_labels = [l for l in labels if l != 0]

            # 1) 直接目标标签
            if desired_label in nz_labels:
                vox = int(lss.GetNumberOfPixels(desired_label))
                mask_bin = sitk.BinaryThreshold(mask_itk, lowerThreshold=desired_label, upperThreshold=desired_label, insideValue=1, outsideValue=0)
                info.update({'strategy': 'exact_label', 'used_label': desired_label, 'voxel_count': vox})
            else:
                # 2) 候选标签优先级
                sel = None
                for cand in alt_candidates:
                    if cand in nz_labels:
                        sel = cand
                        break
                # 3) 退化为“最大体素标签”
                if sel is None and fallback_largest and len(nz_labels) > 0:
                    # 选择体素最多的标签
                    sizes = [(lab, int(lss.GetNumberOfPixels(lab))) for lab in nz_labels]
                    sizes.sort(key=lambda x: x[1], reverse=True)
                    sel = sizes[0][0]
                if sel is not None:
                    vox = int(lss.GetNumberOfPixels(sel))
                    mask_bin = sitk.BinaryThreshold(mask_itk, lowerThreshold=sel, upperThreshold=sel, insideValue=1, outsideValue=0)
                    info.update({'strategy': 'alt_or_largest', 'used_label': int(sel), 'voxel_count': vox})
                else:
                    # 4) 实在不行，非零即ROI
                    mask_bin = sitk.Cast(sitk.NotEqual(mask_itk, 0), sitk.sitkUInt8)
                    vox = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_bin)))
                    info.update({'strategy': 'integer_nonzero', 'used_label': 'nonzero', 'voxel_count': vox})

        else:
            # 浮点掩膜：优先阈值；若为空则非零
            mask_bin, vox, strat = binarize_float_mask(mask_itk, float_threshold, case_id)
            info.update({'strategy': strat, 'used_label': 'float', 'voxel_count': vox, 'found_labels': None})

        # 可选：只保留最大连通域
        if keep_largest and info['voxel_count'] > 0:
            mask_bin = keep_largest_component(mask_bin)
            vox = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_bin)))
            info['voxel_count'] = vox
            info['strategy'] = (info['strategy'] or '') + '+largest'

        # 最终转UInt8
        mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)
        return mask_bin, info

    except Exception as e:
        print(f"[ERROR] 病例 {case_id}: 掩膜清洗失败: {e}")
        return None, {'strategy': 'error', 'used_label': None, 'voxel_count': 0, 'found_labels': None, 'error': str(e)}

# ================== 核心处理流程 ==================

def process_case(image_path: str, mask_path: str, case_id: str, label_val: int,
                 min_voxels_features: int = MIN_VOXELS_FOR_FEATURES):
    """
    对单个病例进行完整的影像组学特征提取流程。
    变更：取消重采样，若几何不匹配则跳过。不再保存ROI图像。
    """
    try:
        # 1) 加载原始数据（.nii.gz）
        image_itk, mask_raw = read_image_and_mask(image_path, mask_path)

        # Debug：原始掩膜标签
        if DEBUG_PRINT_MASK_LABELS:
            print_mask_summary(case_id, mask_raw, prefix="DEBUG-RAW")

        # 2) 检查几何匹配（取消重采样）
        if not check_geometry_match(mask_raw, image_itk):
            print(f"[WARNING] 病例 {case_id}: 掩膜与影像几何不匹配（已取消重采样策略），跳过该病例。")
            return None
        mask_aligned = mask_raw  # 无重采样

        # 3) 清洗/统一掩膜（Label=1）
        mask_clean, info = sanitize_and_unify_mask(
            mask_aligned,
            desired_label=label_val,
            alt_candidates=ALT_LABEL_CANDIDATES,
            fallback_largest=FALLBACK_USE_LARGEST_LABEL,
            float_threshold=FLOAT_MASK_THRESHOLD,
            keep_largest=KEEP_ONLY_LARGEST_COMPONENT,
            case_id=case_id
        )

        if mask_clean is None:
            print(f"[WARNING] 病例 {case_id}: 掩膜清洗失败，跳过（strategy={info.get('strategy')}）")
            return None

        # Debug：清洗后统计
        if DEBUG_PRINT_MASK_LABELS:
            print_mask_summary(case_id, mask_clean, prefix="DEBUG-CLEAN")

        # 4) 体素统计
        voxel_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_clean)))
        if voxel_count == 0:
            # 打印更具体的标签信息帮助定位
            if is_integer_image(mask_aligned):
                lss = sitk.LabelShapeStatisticsImageFilter()
                lss.Execute(mask_aligned)
                labels = sorted([int(x) for x in lss.GetLabels()])
                if labels:
                    msg_lines = [f"[WARNING] 病例 {case_id}: 清洗后掩膜为空，原掩膜可用标签: {labels}"]
                    for lab in labels:
                        msg_lines.append(f"  - Label={lab} Voxels={int(lss.GetNumberOfPixels(lab))}")
                    print("\n".join(msg_lines))
                else:
                    print(f"[WARNING] 病例 {case_id}: 掩膜清洗后为空（无非零体素），跳过。")
            else:
                arr = sitk.GetArrayViewFromImage(mask_aligned)
                print(f"[WARNING] 病例 {case_id}: 浮点掩膜清洗后非零体素={int(np.count_nonzero(arr))}，跳过。")
            return None

        # 5) 若体素太少则不做特征提取
        if voxel_count < int(min_voxels_features):
            print(f"[WARNING] 病例 {case_id}: ROI体素数 ({voxel_count}) 低于阈值({min_voxels_features})，跳过特征提取。")
            return None

        # 6) 构建适合当前病例的特征提取器
        extractor = _build_case_extractor(image_itk)
        if not extractor:
            print(f"[ERROR] 病例 {case_id}: 无法构建特征提取器，跳过。")
            return None

        # 7) 执行特征提取（mask_clean 已统一为 Label=1）
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                result = extractor.execute(image_itk, mask_clean, label=label_val)

        # 8) 清理与格式化结果
        clean_result = {}
        for key, value in result.items():
            if not key.startswith('diagnostics_'):
                clean_result[key] = _to_scalar(value)

        # 9) 汇总
        base_info = {'ID': case_id, 'Modality': modality, 'Label': label_val}
        return {**base_info, **clean_result}

    except Exception as e:
        print(f"[ERROR] 处理病例 {case_id} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================== 主程序 ==================

def main():
    print("开始批量提取影像组学特征...")
    print(f"图像目录: {image_dir}")
    print(f"掩膜目录: {mask_dir}")
    print(f"结果目录: {output_dir}")
    print(f"特征CSV: {output_csv}")

    # 扫描影像文件（.nii.gz）
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        if not image_files:
            raise FileNotFoundError("图像目录中未找到 .nii.gz 文件。")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    all_features = []

    for image_filename in tqdm(image_files, desc="处理病例"):
        case_id = image_filename.replace('.nii.gz', '')
        image_path = os.path.join(image_dir, image_filename)
        mask_path = os.path.join(mask_dir, image_filename) # 假设掩膜与影像同名（均为 .nii.gz）

        if not os.path.exists(mask_path):
            print(f"[WARNING] 找不到病例 {case_id} 的掩膜文件: {mask_path}，已跳过。")
            continue

        features = process_case(
            image_path=image_path,
            mask_path=mask_path,
            case_id=case_id,
            label_val=TARGET_LABEL,
            min_voxels_features=MIN_VOXELS_FOR_FEATURES
        )

        if features:
            all_features.append(features)

    if not all_features:
        print("\n处理完成，但未能成功提取任何病例的特征。请检查输入文件和配置。")
        print("提示：干净掩膜仍已尝试保存，可在输出目录中查看。")
        return

    print("\n所有病例处理完毕，正在保存特征结果...")
    df = pd.DataFrame(all_features)

    # 将标识列移动到最前面
    id_cols = ['ID', 'Modality', 'Label']
    feature_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + sorted(feature_cols)]

    df.to_csv(output_csv, index=False)
    print("\n✅ 结果摘要:")
    print(f"  成功提取特征的病例数: {len(all_features)}")
    print(f"  提取的影像组学特征数 (不含诊断信息): {len(feature_cols)}")
    print(f"  特征CSV保存至: {output_csv}")

    print("\n✅ 批量处理完成！")

if __name__ == "__main__":
    main()
