import os
import re
import numpy as np
import pandas as pd
import pydicom
from glob import glob

# ============ 路径配置 ============
FEATURE_CSV_PATH = r"D:\xiaojuan\V2\radiomics_features_all_casesV2.csv"
DICOM_ROOT_DIR   = r"C:\new1\V3\images"
OUTPUT_DIR       = r"D:\xiaojuan\V2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 批次来源："Auto"（默认，Site 优先）/"Site"/"Scanner"
BATCH_SOURCE = "Auto"
REF_BATCH = None  # 例如设为某个真实存在的站点名

# ============ 文本规范化工具 ============
def _clean_text(x):
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "" or s.lower() in {"unknown", "none", "na", "n/a"}:
        return None
    return s

# ============ 提取 DICOM 元数据 ============
def extract_dicom_metadata(dicom_dir):
    rows, logs = [], []
    patients = [f for f in os.listdir(dicom_dir) if os.path.isdir(os.path.join(dicom_dir, f))]
    try:
        patients = sorted(patients, key=lambda x: int(x))
    except ValueError:
        patients = sorted(patients)

    for folder in patients:
        folder_path = os.path.join(dicom_dir, folder)
        dcm_files = glob(os.path.join(folder_path, "*.dcm")) or glob(os.path.join(folder_path, "*.DCM"))
        if not dcm_files:
            msg = f"[警告] 患者 {folder} 未找到 DICOM 文件（{folder_path}）"
            print(msg); logs.append(msg)
            continue

        try:
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)

            manufacturer  = _clean_text(getattr(ds, "Manufacturer", None))
            model         = _clean_text(getattr(ds, "ManufacturerModelName", None))
            serial        = _clean_text(getattr(ds, "DeviceSerialNumber", None))
            station       = _clean_text(getattr(ds, "StationName", None))

            inst_name     = _clean_text(getattr(ds, "InstitutionName", None))
            inst_dept     = _clean_text(getattr(ds, "InstitutionalDepartmentName", None))
            inst_addr     = _clean_text(getattr(ds, "InstitutionAddress", None))

            scanner = station or serial or model or "Unknown"
            site    = inst_name or inst_dept or inst_addr or "Unknown"

            # 构造 Batch
            if BATCH_SOURCE.lower() == "site":
                batch = site
            elif BATCH_SOURCE.lower() == "scanner":
                batch = scanner
            else:
                combo = None
                if (inst_name or inst_addr) and (manufacturer or model):
                    combo = f"{inst_name or inst_addr}|{manufacturer or ''}|{model or ''}"
                elif manufacturer or model or station:
                    combo = f"{manufacturer or ''}|{model or ''}|{station or ''}"
                batch = site if site and site != "Unknown" else (combo if combo and combo.strip("|") else scanner)

            try:
                patient_id = str(int(folder))
            except ValueError:
                patient_id = folder.strip()

            rows.append({
                "ID": patient_id,
                "Site": site or "Unknown",
                "Scanner": scanner or "Unknown",
                "Manufacturer": manufacturer or "Unknown",
                "Model": model or "Unknown",
                "StationName": station or "Unknown",
                "Batch": (batch or "Unknown")
            })
            logs.append(f"ID={patient_id} | Site={site} | Scanner={scanner} | Batch={batch}")

        except Exception as e:
            msg = f"[错误] 读取患者 {folder} 失败: {e}"
            print(msg); logs.append(msg)

    log_path = os.path.join(OUTPUT_DIR, "metadata_extraction_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    print(f"📝 元数据日志: {log_path}")
    return pd.DataFrame(rows)

# ============ 改进的特征数值化函数 ============
def coerce_features_to_numeric(df, candidate_cols, min_numeric_ratio=0.5):
    """
    彻底改进的特征数值化函数
    """
    df_num = pd.DataFrame(index=df.index)
    kept, dropped = [], []
    
    print(f"\n🔍 开始数值化转换 {len(candidate_cols)} 个候选特征...")
    
    for i, c in enumerate(candidate_cols):
        if i % 50 == 0:
            print(f"  处理进度: {i}/{len(candidate_cols)}")
            
        s = df[c]
        
        # 跳过全为空的列
        if s.isna().all():
            dropped.append((c, "全为空值"))
            continue
            
        # 记录原始数据类型
        orig_dtype = s.dtype
        orig_sample = s.head(3).tolist() if len(s) > 0 else []
        
        try:
            # 方法1: 直接转换
            parsed = pd.to_numeric(s, errors='coerce')
            numeric_count = parsed.notna().sum()
            numeric_ratio = numeric_count / len(s)
            
            # 如果直接转换效果不好，尝试字符串清理
            if numeric_ratio < min_numeric_ratio:
                # 方法2: 字符串清理后转换
                s_clean = s.astype(str).str.replace(',', '.', regex=False)
                s_clean = s_clean.str.replace(r'[^\d\.\-eE]', '', regex=True)
                s_clean = s_clean.replace(['', 'nan', 'None', 'null', 'NA'], np.nan)
                parsed = pd.to_numeric(s_clean, errors='coerce')
                numeric_count = parsed.notna().sum()
                numeric_ratio = numeric_count / len(s)
            
            # 检查是否满足保留条件
            if numeric_ratio >= min_numeric_ratio and numeric_count > 0:
                df_num[c] = parsed
                kept.append(c)
                print(f"  ✅ 保留: {c} (转换率: {numeric_ratio:.3f}, 原始类型: {orig_dtype}, 样本: {orig_sample})")
            else:
                dropped.append((c, f"数值比例过低: {numeric_ratio:.3f}"))
                print(f"  ❌ 丢弃: {c} (转换率: {numeric_ratio:.3f}, 原始类型: {orig_dtype})")
                
        except Exception as e:
            dropped.append((c, f"转换异常: {str(e)}"))
            print(f"  ❌ 异常: {c} - {str(e)}")
    
    print(f"\n📊 数值化结果: 保留 {len(kept)} 个, 丢弃 {len(dropped)} 个")
    return df_num, kept, dropped

# ============ 主流程 ============
def main():
    print("🔍 步骤1: 读取特征 CSV ...")
    df_feat = pd.read_csv(FEATURE_CSV_PATH)

    if "ID" not in df_feat.columns:
        raise KeyError("特征 CSV 必须包含 'ID' 列")

    # 清理 CSV 中可能的杂项列
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]
    drop_like = [c for c in df_feat.columns if str(c).startswith("Unnamed:")]
    if drop_like:
        df_feat = df_feat.drop(columns=drop_like)

    # 规范 ID
    df_feat["ID"] = df_feat["ID"].astype(str).apply(lambda x: re.sub(r"\.nii(\.gz)?$", "", x.strip(), flags=re.IGNORECASE))
    print(f"✅ 加载特征：{len(df_feat)} 样本，{len(df_feat.columns)-1} 个特征（不含 ID）")
    
    # 调试：显示前几行数据
    print("\n📋 特征数据前3行:")
    print(df_feat.head(3))
    print(f"\n📋 特征列名 ({len(df_feat.columns)} 列):")
    for i, col in enumerate(df_feat.columns):
        print(f"  {i:3d}: {col}")

    print("📁 步骤2: 提取 DICOM 元数据 ...")
    df_meta = extract_dicom_metadata(DICOM_ROOT_DIR)
    if df_meta.empty:
        raise RuntimeError("未能从 DICOM 提取元数据，请检查路径或文件。")
    print(f"✅ 提取元数据：{len(df_meta)} 位患者")

    print("🔗 步骤3: 合并特征与元数据 ...")
    df = df_feat.merge(df_meta, on="ID", how="left")

    for col in ["Site", "Scanner", "Batch"]:
        df[col] = df[col].fillna("Unknown")
        df.loc[df[col].astype(str).str.strip().eq(""), col] = "Unknown"

    merged_csv = os.path.join(OUTPUT_DIR, "radiomics_features_with_site.csv")
    df.to_csv(merged_csv, index=False)
    print(f"💾 已保存：{merged_csv}")

    # 打印分布
    print("\n📊 批次（Batch）分布：")
    print(df["Batch"].value_counts().head(20))
    print("\n📊 Site 分布：")
    print(df["Site"].value_counts().head(20))
    print("\n📊 Scanner 分布：")
    print(df["Scanner"].value_counts().head(20))

    # ============ 步骤4: 识别特征列并数值化 ============
    meta_cols = {"ID", "Site", "Scanner", "Manufacturer", "Model", "StationName", "Batch", "Label", "Modality"}
    candidate_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"\n🎯 识别到 {len(candidate_cols)} 个候选特征列:")
    for col in candidate_cols[:10]:  # 只显示前10个
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"  {col}: {sample_vals} (类型: {df[col].dtype})")

    # 使用改进的数值化函数，降低阈值以确保更多特征被保留
    df_num, feature_cols, dropped = coerce_features_to_numeric(df, candidate_cols, min_numeric_ratio=0.5)

    # 记录被丢弃列
    dropped_report = os.path.join(OUTPUT_DIR, "dropped_feature_columns.txt")
    with open(dropped_report, "w", encoding="utf-8") as f:
        for name, reason in dropped:
            f.write(f"{name}\t{reason}\n")

    kept_report = os.path.join(OUTPUT_DIR, "kept_feature_columns.txt")
    with open(kept_report, "w", encoding="utf-8") as f:
        for name in feature_cols:
            f.write(f"{name}\n")

    print(f"\n🧾 识别到数值特征列：{len(feature_cols)} 个（清单见 {kept_report}）")
    print(f"🧾 被丢弃候选列：{len(dropped)} 个（原因见 {dropped_report}）")

    if len(feature_cols) == 0:
        print("❌ 错误: 未识别到任何数值特征列!")
        print("可能的原因:")
        print("1. 特征CSV文件格式问题")
        print("2. 特征值包含大量非数值字符") 
        print("3. 特征列名识别错误")
        print("4. 数据确实全为空值")
        
        # 详细诊断
        print("\n🔍 详细诊断信息:")
        for col in candidate_cols[:5]:  # 检查前5个候选列
            sample_data = df[col].dropna().head(5).tolist()
            print(f"列 '{col}': 样本值 = {sample_data}, 类型 = {df[col].dtype}")
        
        raise RuntimeError("未识别到任何数值特征列。请检查 dropped_feature_columns.txt 以定位原因。")

    # ============ 步骤5: 缺失值处理 ============
    print(f"\n🔧 步骤5: 缺失值处理...")
    X = df_num[feature_cols].to_numpy(dtype=float)
    
    # 检查数据质量
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  缺失值数量: {nan_count}")
    print(f"  无穷值数量: {inf_count}")
    
    # 处理 NaN/inf
    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = np.nan
        print("  ⚠️ 检测到非有限值，已转换为NaN")
    
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        # 对于全为NaN的列，用0填充
        col_mean[np.isnan(col_mean)] = 0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        print(f"  ✅ 已用列均值填充 {np.isnan(X).sum()} 个缺失值")

    # ============ 步骤6: ComBat（neuroHarmonize） ============
    print(f"\n🔧 步骤6: ComBat 批次校正...")
    batch_series = df["Batch"].astype(str).fillna("Unknown")
    n_batches = batch_series.nunique(dropna=False)
    
    if n_batches < 2:
        print("ℹ️ 仅检测到 1 个批次，跳过 ComBat，直接保存特征。")
        X_adj = X.copy()
    else:
        print(f"  检测到 {n_batches} 个批次，进行 ComBat 校正...")
        # 确保参考批次有效
        ref = REF_BATCH if (REF_BATCH is not None and REF_BATCH in set(batch_series)) else None
        if REF_BATCH is not None and ref is None:
            print(f"  ⚠️ 指定的 REF_BATCH='{REF_BATCH}' 不在数据中，已自动改为 None。")

        covars = pd.DataFrame({"batch": batch_series})
        try:
            from neuroHarmonize import harmonizationLearn
            model, X_adj = harmonizationLearn(X, covars, ref_batch=ref, eb=True)
            print("  ✅ ComBat 校正完成")
        except Exception as e:
            print(f"  ❌ ComBat 校正失败: {e}")
            print("  ⚠️ 使用原始数据进行后续处理")
            X_adj = X.copy()

    # ============ 步骤7: 结果落盘 ============
    print(f"\n💾 步骤7: 保存结果...")
    out_cols_first = [c for c in ["ID", "Site", "Scanner", "Batch"] if c in df.columns]
    df_out = pd.concat(
        [df[out_cols_first].reset_index(drop=True),
         pd.DataFrame(X_adj, columns=feature_cols)],
        axis=1
    )
    
    # 验证输出数据
    print(f"  输出数据形状: {df_out.shape}")
    print(f"  特征列数量: {len(feature_cols)}")
    print(f"  前3个特征列样本值:")
    for col in feature_cols[:3]:
        sample_vals = df_out[col].head(3).tolist()
        print(f"    {col}: {sample_vals}")

    out_csv = os.path.join(OUTPUT_DIR, "radiomics_features_combat_harmonized.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"✅ 完成！已保存：{out_csv}")
    
    # 最终验证
    final_check = pd.read_csv(out_csv)
    feature_data_present = final_check[feature_cols].notna().any().any()
    if feature_data_present:
        print("🎉 验证通过: 输出文件包含有效的特征数据!")
    else:
        print("❌ 警告: 输出文件中的特征数据可能仍为空，请检查输入数据格式!")

if __name__ == "__main__":
    main()