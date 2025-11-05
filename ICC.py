import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 设置文件路径 ---
file_path_before = r"D:\xiaojuan\A2\CT_radiomics\radiomics_features_all_casesICCA前.csv"
file_path_after = r"D:\xiaojuan\A2\CT_radiomics\radiomics_features_all_casesICCA后.csv"

# --- 2. 加载数据 ---
try:
    df_before = pd.read_csv(file_path_before)
    df_after = pd.read_csv(file_path_after)
except FileNotFoundError:
    print("错误：无法找到文件。请检查您的文件路径是否正确。")
    exit()

# --- 3. 数据预处理与合并 ---

# 重命名ID列
id_column_name = df_before.columns[0]
df_before = df_before.rename(columns={id_column_name: 'ID'})
df_after = df_after.rename(columns={id_column_name: 'ID'})

# 确保ID为字符串类型，便于合并
df_before['ID'] = df_before['ID'].astype(str)
df_after['ID'] = df_after['ID'].astype(str)

# 找出两个数据集中共有的ID
common_ids = pd.merge(df_before[['ID']], df_after[['ID']], on='ID')['ID'].unique()

# 过滤只保留共有ID的行
df_before = df_before[df_before['ID'].isin(common_ids)].copy()
df_after = df_after[df_after['ID'].isin(common_ids)].copy()

# 添加测量时间标识
df_before['measurement'] = 'ICCV前'
df_after['measurement'] = 'ICCV后'

# 合并数据
df_long = pd.concat([df_before, df_after], ignore_index=True)

# 确保所有特征列为数值类型
feature_columns = [col for col in df_before.columns if col not in ['ID', 'measurement']]
for col in feature_columns:
    df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

# --- 4. 循环计算每个特征的 ICC ---

icc_results = []

print("正在计算每个特征的 ICC 值...")
for feature in feature_columns:
    # 提取当前特征的数据
    icc_data = df_long[['ID', 'measurement', feature]].dropna()

    # 检查每个ID是否都有两次测量
    id_counts = icc_data.groupby('ID').size()
    valid_ids = id_counts[id_counts == 2].index
    if len(valid_ids) < 1:
        print(f"警告：特征 '{feature}' 有效样本不足，跳过。")
        continue

    icc_data = icc_data[icc_data['ID'].isin(valid_ids)]

    # 检查特征值是否全部相同（方差为零）
    if icc_data[feature].nunique() <= 1:
        print(f"警告：特征 '{feature}' 的测量值无变化，无法计算ICC。")
        continue

    # 计算ICC
    try:
        icc = pg.intraclass_corr(
            data=icc_data,
            targets='ID',
            raters='measurement',
            ratings=feature,
            nan_policy='omit'
        )

        # 使用 ICC3 (ICC(3,1)) —— 固定测量者、绝对一致性
        icc_row = icc.set_index('Type').loc['ICC3']
        icc_value = icc_row['ICC']
        ci95_low, ci95_high = icc_row['CI95%']

        # 检查ICC值是否为NaN或inf
        if pd.isna(icc_value) or np.isinf(icc_value):
            print(f"警告：特征 '{feature}' 的ICC值为NaN或inf，跳过。")
            continue

        icc_results.append({
            'Feature': feature,
            'ICC_Type': 'ICC3',
            'ICC_Value': icc_value,
            'CI95_Low': ci95_low,
            'CI95_High': ci95_high
        })

    except Exception as e:
        print(f"计算特征 '{feature}' 时出错: {e}")
        continue

# 转换为DataFrame
results_df = pd.DataFrame(icc_results)

if results_df.empty:
    print("错误：未成功计算任何特征的 ICC，请检查数据格式或内容。")
    exit()

# --- 5. 显示并保存结果 ---

print("\n" + "="*70 + "\n")
print("--- ICC 计算结果（前10个特征）---")
print(results_df.head(10))
print(f"\n... 总共计算了 {len(results_df)} 个特征的 ICC。")
print("="*70 + "\n")

output_path = r"D:\xiaojuan\A2\CT_radiomics\ICC_resultsA.csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 每个特征的 ICC 结果已保存至：{output_path}")

# --- 6. 整体ICC汇总 ---

mean_icc = results_df['ICC_Value'].mean()
median_icc = results_df['ICC_Value'].median()
std_icc = results_df['ICC_Value'].std()

print("\n" + "="*50)
print("--- 整体 ICC 汇总（影像组学标准报告）---")
print(f"平均 ICC (Mean ICC3):   {mean_icc:.6f}")
print(f"中位数 ICC (Median ICC3): {median_icc:.6f}")
print(f"标准差 (SD):           {std_icc:.6f}")
print("="*50 + "\n")

summary_data = {
    'Metric': ['Mean ICC3', 'Median ICC3', 'SD of ICC3'],
    'Value': [mean_icc, median_icc, std_icc]
}
summary_df = pd.DataFrame(summary_data)
summary_output_path = r"D:\xiaojuan\A2\CT_radiomics\ICC_summaryA.csv"
summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
print(f"✅ 整体 ICC 汇总已保存至：{summary_output_path}")

# --- 7. 可靠性分级（Landis & Koch 标准）---

def classify_reliability(icc_val):
    if icc_val >= 0.80:
        return "Almost Perfect"
    elif icc_val >= 0.60:
        return "Substantial"
    elif icc_val >= 0.40:
        return "Moderate"
    elif icc_val >= 0.20:
        return "Fair"
    elif icc_val >= 0.00:
        return "Slight"
    else:
        return "Poor"

results_df['Reliability'] = results_df['ICC_Value'].apply(classify_reliability)
reliability_summary = results_df['Reliability'].value_counts().reindex([
    "Almost Perfect", "Substantial", "Moderate", "Fair", "Slight", "Poor"
]).fillna(0).astype(int)

print("\n--- ICC 可靠性等级分布（Landis & Koch, 1977）---")
print(reliability_summary)
print(f"\n🌟 高可重复性特征（ICC ≥ 0.8）数量：{reliability_summary.get('Almost Perfect', 0)} / {len(results_df)}")

reliability_output_path = r"D:\xiaojuan\A2\CT_radiomics\ICC_reliability_distributionA.csv"
reliability_summary.to_frame(name='Count').to_csv(reliability_output_path, encoding='utf-8-sig')
print(f"✅ 可靠性分布已保存至：{reliability_output_path}")

# --- 8. 可视化 ICC 分布 ---

plt.figure(figsize=(10, 6))
sns.histplot(results_df['ICC_Value'], bins=30, kde=True, color='#4B8BBE', edgecolor='black', alpha=0.8)
plt.title('Distribution of ICC(3,1) Across All Radiomic Features', fontsize=14, fontweight='bold')
plt.xlabel('ICC Value', fontsize=12)
plt.ylabel('Number of Features', fontsize=12)
plt.axvline(mean_icc, color='red', linestyle='--', linewidth=2, label=f'Mean ICC = {mean_icc:.3f}')
plt.axvline(0.8, color='green', linestyle='-.', linewidth=2, label='Threshold (ICC ≥ 0.8)')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(-0.1, 1.05)

fig_output_path = r"D:\xiaojuan\A2\CT_radiomics\ICC_distribution_plotA.png"
plt.savefig(fig_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ ICC 分布图已保存至：{fig_output_path}")