import os
import sys
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ===== 用户路径设置 =====
INPUT_CSV = r"/home/featurize/data/A2/radiomicsVallselected.csv"
OUTPUT_DIR = r"/home/featurize/data/A2/机器学习模型V2"

RANDOM_STATE = 42
N_BOOTSTRAPS = 2000
AUTO_INSTALL_MISSING = True

np.random.seed(RANDOM_STATE)

# ===== 工具函数 =====
def safe_makedirs(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: 创建目录失败，但继续运行: {path}. Error: {e}")

def try_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "ansi", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法读取CSV，请检查文件编码或路径: {path}\n最后错误: {last_err}")

def normalize_group_col(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.lower()
    m = m.replace({
        "training": "train", "tr": "train",
        "validation": "val", "valid": "val", "va": "val",
        "testing": "test", "te": "test"
    })
    return m

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_bootstraps: int = 2000, seed: int = 42) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if len(aucs) < 10:
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
        return (auc, np.nan, np.nan)
    auc = roc_auc_score(y_true, y_score)
    low = float(np.percentile(aucs, 2.5))
    high = float(np.percentile(aucs, 97.5))
    return (auc, low, high)

def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        mask = np.isfinite(thresholds)
        if mask.sum() == 0:
            return 0.5
        fpr, tpr, thresholds = fpr[mask], tpr[mask], thresholds[mask]
        j = tpr - fpr
        idx = int(np.argmax(j))
        thr = float(thresholds[idx])
        if np.isnan(thr) or np.isinf(thr):
            return 0.5
        return thr
    except Exception:
        return 0.5

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc, ci_low, ci_high = bootstrap_auc_ci(y_true, y_score, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_STATE)
    except Exception:
        auc, ci_low, ci_high = (np.nan, np.nan, np.nan)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    return {
        "Accuracy": acc,
        "AUC": (auc if not np.isnan(auc) else np.nan),
        "95% CI": f"($${ci_low:.3f}$$-$${ci_high:.3f}$$)" if (ci_low==ci_low and ci_high==ci_high) else "NA",
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Threshold": threshold,
        "MCC": mcc
    }

def get_proba_for_positive(estimator, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    proba = estimator.predict_proba(X)
    classes_ = getattr(estimator, "classes_", None)
    if classes_ is None:
        try:
            clf = estimator.named_steps["clf"]
            classes_ = getattr(clf, "classes_", None)
        except Exception:
            classes_ = None

    if classes_ is None:
        classes_ = np.array([0, 1])
    else:
        if isinstance(classes_, list):
            classes_ = np.array(classes_)

    pos_indices = np.where(classes_ == 1)[0]
    if len(pos_indices) == 0:
        pos_idx = 1 if proba.shape[1] > 1 else 0
    else:
        pos_idx = int(pos_indices[0])
    prob1 = proba[:, pos_idx]
    if proba.shape[1] == 2:
        neg_idx = 1 - pos_idx if pos_idx in [0, 1] else (0 if pos_idx != 0 else 1)
        prob0 = proba[:, neg_idx]
    else:
        prob0 = 1.0 - prob1
    return prob0, prob1

def fmt_float(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    try:
        return f"{x:.3f}"
    except Exception:
        return str(x)

def safe_install(pkg: str) -> bool:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--upgrade", "-q"])
        return True
    except Exception as e:
        print(f"Warning: 自动安装 {pkg} 失败，跳过该模型。错误: {e}")
        return False

def get_xgb_cls():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        if AUTO_INSTALL_MISSING and safe_install("xgboost"):
            try:
                from xgboost import XGBClassifier
                return XGBClassifier
            except Exception:
                return None
        return None

def plot_roc_three_sets(model_name: str,
                        y_train, p_train,
                        y_val, p_val,
                        y_test, p_test,
                        out_path: str):
    plt.figure(figsize=(7, 6), dpi=150)
    
    def plot_one(y, p, color, label_prefix):
        if y is None or len(y) == 0 or len(np.unique(y)) < 2 or p is None:
            return
        try:
            fpr, tpr, _ = roc_curve(y, p)
            auc, low, high = bootstrap_auc_ci(y, p, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_STATE)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"{label_prefix} AUC=$${fmt_float(auc)}$$ (95% CI: $${fmt_float(low)}$$-$${fmt_float(high)}$$)")
        except Exception as e:
            print(f"Warning: 绘制ROC失败 {model_name} {label_prefix}: {e}")

    plt.plot([0, 1], [0, 1], color='gray', lw=1, ls='--')
    
    plot_one(y_train, p_train, 'red', 'Train')
    plot_one(y_val, p_val, 'blue', 'Validation')
    plot_one(y_test, p_test, 'green', 'Test')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(out_path)
    except Exception as e:
        print(f"Warning: 保存ROC图失败: {out_path}. Error: {e}")
    plt.close()

# ===== 主流程 =====
def main():
    safe_makedirs(OUTPUT_DIR)

    # 读数据
    df = try_read_csv(INPUT_CSV)
    expected_cols = ["ID", "label", "group"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV中缺少必要列: {missing}. 需包含列: {expected_cols}")

    # 基本清洗
    df = df.copy()
    df["group"] = normalize_group_col(df["group"])
    df["ID"] = df["ID"].astype(str)
    for c in ["label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df["label"].isna()].copy()
    df["label"] = df["label"].astype(int)

    # 划分数据集
    df_train = df[df["group"] == "train"].copy()
    df_val   = df[df["group"] == "val"].copy()
    df_test  = df[df["group"] == "test"].copy()

    if len(df_train) == 0:
        raise RuntimeError("训练集为空（group=='train'）。请检查group列。")
    if len(df_val) == 0:
        print("Warning: 验证集为空（group=='val'），将使用阈值0.5。")
    if len(df_test) == 0:
        print("Warning: 测试集为空（group=='test'）。")

    # 取消临床特征，仅保留组学特征
    feature_cols = [c for c in df.columns if c not in ["ID", "label", "group"]]
    numeric_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            print(f"⚠️ 特征 '{col}' 非数值型，将被自动丢弃。")
    if not numeric_features:
        raise RuntimeError("没有找到任何数值型特征！请检查CSV文件。")

    features_num = numeric_features
    print(f"✅ 识别到 {len(features_num)} 个数值型组学特征:")
    for i, col in enumerate(features_num[:10]):
        print(f"  {i+1:2d}: {col}")

    # 提取特征和标签
    X_train = df_train[features_num]
    y_train = df_train["label"].values
    X_val   = df_val[features_num] if len(df_val) > 0 else pd.DataFrame(columns=features_num)
    y_val   = df_val["label"].values if len(df_val) > 0 else np.array([], dtype=int)
    X_test  = df_test[features_num] if len(df_test) > 0 else pd.DataFrame(columns=features_num)
    y_test  = df_test["label"].values if len(df_test) > 0 else np.array([], dtype=int)

    # 预处理
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(threshold=0.0))
            ]), features_num)
        ],
        remainder="drop"
    )

    # 模型定义
    models = {}
    
    # XGBoost
    XGBClassifier = get_xgb_cls()
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(steps=[
            ("pre", preprocessor_tree),
            ("clf", XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0,
                reg_alpha=1.0, random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="logloss", tree_method="hist"
            ))
        ])
    else:
        models["XGBoost"] = None


    # 超参数搜索空间
    param_distributions = {
        "XGBoost": {
            "clf__n_estimators": [150, 200, 300],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__min_child_weight": [5, 7, 10],
            "clf__subsample": [0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.7, 0.8, 0.9],
            "clf__reg_lambda": [5, 8, 10],
            "clf__reg_alpha": [0.5, 1.0, 2.0],
        },
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # 逐模型训练
    for name, base_pipe in models.items():
        print(f"\n==== 训练与评估: {name} ====")
        metrics_rows = []
        probs_rows = []

        metrics_csv = os.path.join(OUTPUT_DIR, f"metrics_{name}.csv")
        probs_csv   = os.path.join(OUTPUT_DIR, f"probs_{name}.csv")
        roc_png     = os.path.join(OUTPUT_DIR, f"roc_{name}.png")

        if base_pipe is None:
            print(f"模型{name}缺少依赖或未能创建，将输出NA占位文件。")
            for set_name in ["train", "val", "test"]:
                metrics_rows.append({
                    "Set": set_name, "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                    "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                    "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": "NA", "MCC": "NA"
                })
            pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=["ID", "Prob_Class0", "Prob_Class1", "TrueLabel", "group"]).to_csv(
                probs_csv, index=False, encoding="utf-8-sig")
            plt.figure()
            plt.title("ROC Curves Comparison")
            plt.savefig(roc_png)
            plt.close()
            continue

        best_pipe = None

        # General models: Use RandomizedSearchCV or default training
        search_space = param_distributions.get(name, {})
        fit_params = {}
        
        # Handle XGBoost's scale_pos_weight
        if name == "XGBoost":
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
            try:
                base_pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
            except Exception:
                pass

        try:
            if search_space:
                search = RandomizedSearchCV(
                    estimator=base_pipe,
                    param_distributions=search_space,
                    n_iter=20,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    refit=True,
                    verbose=0,
                    error_score='raise'
                )
                search.fit(X_train, y_train, **fit_params)
                best_pipe = search.best_estimator_
                print(f"  -> 最优参数: {search.best_params_}")
                print(f"  -> 训练CV最佳AUC: $${search.best_score_:.3f}$$")
            else: # If no search space is defined for XGBoost, it would fall here, but we have one.
                base_pipe.fit(X_train, y_train, **fit_params)
                best_pipe = base_pipe
                print("  -> 已使用默认参数训练")
        except Exception as e:
            print(f"Warning: {name} 训练/调参失败，将输出NA占位文件。错误: {e}")
            best_pipe = None

        # 如果所有训练方式都失败
        if best_pipe is None:
            print(f"  -> 模型 {name} 无法训练，输出NA文件。")
            for set_name in ["train", "val", "test"]:
                metrics_rows.append({
                    "Set": set_name, "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                    "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                    "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": "NA", "MCC": "NA"
                })
            pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=["ID", "Prob_Class0", "Prob_Class1", "TrueLabel", "group"]).to_csv(
                probs_csv, index=False, encoding="utf-8-sig")
            plt.figure()
            plt.title("ROC Curves Comparison")
            plt.savefig(roc_png)
            plt.close()
            continue

        # 概率校准 (applies to XGBoost as it's not LR or NB)
        if len(y_val) > 0: # Check if validation set is available for calibration
            try:
                # Need to use best_pipe's actual estimator for prefit
                # CalibratedClassifierCV requires a classifier that can predict_proba
                # and doesn't explicitly run pre-processing steps.
                # So we train a CalibratedClassifierCV on the output of the full pipeline.
                
                # First, get the preprocessed validation data
                X_val_calib = best_pipe.named_steps['pre'].transform(X_val)
                
                # Then, instantiate CalibratedClassifierCV with the internal classifier
                # and fit it on the preprocessed data and y_val
                calibrator_base_clf = best_pipe.named_steps['clf']
                calibrator = CalibratedClassifierCV(calibrator_base_clf, cv="prefit", method="sigmoid")
                
                # Fit the calibrator on the preprocessed data and original labels
                calibrator.fit(X_val_calib, y_val)
                
                # Create a new pipeline with the calibrated classifier
                # This ensures the preprocessing is still part of the pipeline when predicting
                best_pipe_calibrated = Pipeline(steps=[
                    ("pre", best_pipe.named_steps['pre']), # Keep the original preprocessor
                    ("clf", calibrator) # Replace the original clf with the calibrated one
                ])
                best_pipe = best_pipe_calibrated # Update best_pipe to the calibrated version
                
                print("  -> 已对模型进行概率校准（Platt scaling）")
            except Exception as e:
                print(f"Warning: 概率校准失败，已跳过。错误: {e}")

        # 计算阈值
        if len(y_val) > 0 and len(np.unique(y_val)) >= 2:
            try:
                _, p1_val = get_proba_for_positive(best_pipe, X_val)
                thr = youden_threshold(y_val, p1_val)
            except Exception:
                thr = 0.5
        else:
            thr = 0.5

        def add_set(set_name, df_set, X_set, y_set):
            nonlocal metrics_rows, probs_rows
            if len(X_set) == 0:
                metrics_rows.append({
                    "Set": set_name, "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                    "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                    "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": fmt_float(thr), "MCC": "NA"
                })
                return

            try:
                p0, p1 = get_proba_for_positive(best_pipe, X_set)
            except Exception as e:
                print(f"Warning: {name} {set_name} 预测失败，将写NA指标。错误: {e}")
                metrics_rows.append({
                    "Set": set_name, "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                    "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                    "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": fmt_float(thr), "MCC": "NA"
                })
                return

            m = compute_all_metrics(y_set, p1, thr)
            m["Set"] = set_name
            metrics_rows.append(m)

            tmp = pd.DataFrame({
                "ID": df_set["ID"].values,
                "Prob_Class0": p0,
                "Prob_Class1": p1,
                "TrueLabel": y_set,
                "group": set_name
            })
            probs_rows.append(tmp)

        # 评估各数据集
        add_set("train", df_train, X_train, y_train)
        if len(df_val) > 0:
            add_set("val", df_val, X_val, y_val)
        else:
            metrics_rows.append({
                "Set": "val", "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": "NA", "MCC": "NA"
            })
        if len(df_test) > 0:
            add_set("test", df_test, X_test, y_test)
        else:
            metrics_rows.append({
                "Set": "test", "Accuracy": "NA", "AUC": "NA", "95% CI": "NA",
                "Sensitivity": "NA", "Specificity": "NA", "PPV": "NA", "NPV": "NA",
                "Precision": "NA", "Recall": "NA", "F1": "NA", "Threshold": "NA", "MCC": "NA"
            })

        # 保存结果
        metrics_df = pd.DataFrame(metrics_rows)
        cols_order = ["Set", "Accuracy", "AUC", "95% CI", "Sensitivity", "Specificity",
                      "PPV", "NPV", "Precision", "Recall", "F1", "Threshold", "MCC"]
        for c in cols_order:
            if c not in metrics_df.columns:
                metrics_df[c] = "NA"
        metrics_df = metrics_df[cols_order]
        for c in cols_order:
            if c in metrics_df.columns:
                metrics_df[c] = metrics_df[c].apply(lambda x: fmt_float(x) if not isinstance(x, str) else x)
        try:
            metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Warning: 保存指标CSV失败: {metrics_csv}. Error: {e}")

        # 保存概率
        if probs_rows:
            probs_df = pd.concat(probs_rows, axis=0, ignore_index=True)
        else:
            probs_df = pd.DataFrame(columns=["ID", "Prob_Class0", "Prob_Class1", "TrueLabel", "group"])
        try:
            probs_df.to_csv(probs_csv, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Warning: 保存概率CSV失败: {probs_csv}. Error: {e}")

        # 绘制ROC
        try:
            p1_train = get_proba_for_positive(best_pipe, X_train)[1] if len(X_train) > 0 else None
            p1_val = get_proba_for_positive(best_pipe, X_val)[1] if len(X_val) > 0 else None
            p1_test = get_proba_for_positive(best_pipe, X_test)[1] if len(X_test) > 0 else None
            
            plot_roc_three_sets(
                name,
                y_train if len(y_train) > 0 else None, p1_train,
                y_val if len(y_val) > 0 else None, p1_val,
                y_test if len(y_test) > 0 else None, p1_test,
                roc_png
            )
        except Exception as e:
            print(f"Warning: 绘制ROC失败: {name}. Error: {e}")

    print("\n✅ 全部完成！结果已保存至：", OUTPUT_DIR)

if __name__ == "__main__":
    main()
