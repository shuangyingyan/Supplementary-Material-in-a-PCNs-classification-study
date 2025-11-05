import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, recall_score,
                             precision_score, f1_score, accuracy_score,
                             matthews_corrcoef)
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1️⃣ 超參數與裝置
MAX_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
DROP_OUT = 0.2          # 改为较低的 dropout，提升信息保留
PATIENCE = 10
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2️⃣ 路徑與常數
IMG_ROOT_ALL = r'/home/featurize/2DA_custom/2D_0'
TRAIN_TXT = r'/home/featurize/train2D.txt'
VAL_TXT = r'/home/featurize/val2D.txt'
TEST_TXT = r'/home/featurize/test2D.txt'
RADIOMICS_CSV = r'/home/featurize/2DA_custom/radiomicsAallselectedPNG.csv'
PRETRAINED_MODEL_PATH = r'/home/featurize/best_resnet50_modelA7.pth'

# 新增辅助函数：构建 image_id → (label, group) 映射
def build_id_mapping():
    """
    从 TRAIN_TXT, VAL_TXT, TEST_TXT 构建 id 到 label 和 group 的映射
    """
    id_to_info = {}

    def add_from_file(txt_path, group_name):
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found.")
            return
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img_id, label = parts
                try:
                    label = int(label)
                except Exception:
                    continue
                if img_id in id_to_info:
                    print(f"Duplicate image_id detected: {img_id}")
                id_to_info[img_id] = {'label': label, 'group': group_name.lower()}
    add_from_file(TRAIN_TXT, "train")
    add_from_file(VAL_TXT, "val")
    add_from_file(TEST_TXT, "test")
    return id_to_info


# 3️⃣ Augmentation
class MedicalAugmentation:
    def __init__(self, image_size=224):
        self.val_test_aug = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


# 4️⃣ 特征提取 Dataset
class FeatureExtractionDataset(Dataset):
    def __init__(self, txt_path, img_root, image_size=224):
        self.img_root = img_root
        self.aug = MedicalAugmentation(image_size).val_test_aug
        self.image_paths = []
        self.image_ids = []

        if not os.path.exists(txt_path):
            print(f"WARNING: Text file not found at '{txt_path}'.")
            return

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                img_id = parts[0]
                img_path = os.path.join(img_root, img_id)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.image_ids.append(img_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.aug(img), self.image_ids[idx]


# 5️⃣ 提取 deep features
def extract_deep_features(model, txt_path, img_root, split_name):
    dataset = FeatureExtractionDataset(txt_path, img_root)
    if len(dataset) == 0:
        return pd.DataFrame()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_feats = []
    all_ids = []

    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc=f"Extracting Deep Features - {split_name}", leave=False):
            imgs = imgs.to(DEVICE)
            feats = model.forward_features(imgs)  # [B, 2048]
            all_feats.append(feats.cpu().numpy())
            all_ids.extend(ids)

    feats_np = np.concatenate(all_feats, axis=0).astype(np.float32)  # 强制 float32
    df = pd.DataFrame(feats_np, columns=[f"deep_f{i+1}" for i in range(feats_np.shape[1])])
    df.insert(0, "image_id", all_ids)
    return df


# 6️⃣ 融合 deep + radiomics 特征（改进：标准化、缺失值处理、方差筛选）
def load_and_merge_features(model):
    # Step 1: 提取 deep features
    deep_train = extract_deep_features(model, TRAIN_TXT, IMG_ROOT_ALL, "Train")
    deep_val = extract_deep_features(model, VAL_TXT, IMG_ROOT_ALL, "Valid")
    deep_test = extract_deep_features(model, TEST_TXT, IMG_ROOT_ALL, "Test")

    for name, df in [('Train', deep_train), ('Val', deep_val), ('Test', deep_test)]:
        if df is None or df.empty:
            raise ValueError(f"{name} deep features are empty. Check paths and files.")

    deep_all = pd.concat([deep_train, deep_val, deep_test], ignore_index=True)

    # Step 2: 加载 radiomics CSV
    if not os.path.exists(RADIOMICS_CSV):
        raise FileNotFoundError(f"Radiomics CSV not found: {RADIOMICS_CSV}")
    rad_df = pd.read_csv(RADIOMICS_CSV)
    id_col = rad_df.columns[0]
    feature_cols = rad_df.columns[1:]

    rad_df = rad_df.rename(columns={id_col: "image_id"})
    rad_features = rad_df[["image_id"] + list(feature_cols)].copy()
    rad_features = rad_features.rename(columns={f: f"rad_{f}" for f in feature_cols})

    # Step 3: 添加 label 和 group
    id_to_info = build_id_mapping()
    if not id_to_info:
        raise ValueError("No valid samples loaded from TXT files. Check format: image_id label")

    info_df = pd.DataFrame.from_dict(id_to_info, orient='index').reset_index()
    info_df = info_df.rename(columns={'index': 'image_id'})
    info_df['image_id'] = info_df['image_id'].astype(str)

    rad_with_label = pd.merge(rad_features, info_df, on="image_id", how="inner")
    if rad_with_label.empty:
        raise ValueError("No overlap between radiomics features and image IDs in TXT files.")

    # Step 4: 合并 deep + radiomics
    merged = pd.merge(deep_all, rad_with_label, on="image_id", how="inner")
    if merged.empty:
        raise ValueError("Merged dataset is empty. Check image_id matching across all sources.")

    # 确保所有浮点列为 float32
    float_cols = merged.select_dtypes(include=['float64']).columns
    merged[float_cols] = merged[float_cols].astype(np.float32)

    print(f"✅ Merged feature shape: {merged.shape} (deep + radiomics)")

    # 分组
    train_df = merged[merged["group"] == "train"].copy().reset_index(drop=True)
    val_df = merged[merged["group"] == "val"].copy().reset_index(drop=True)
    test_df = merged[merged["group"] == "test"].copy().reset_index(drop=True)

    # 类别分布
    print("\nLabel distribution after merging:")
    print(merged.groupby(['group', 'label']).size())

    # Step 5: 缺失值和极值处理、并进行标准化与特征筛选
    # 统一的 feature 列（除 image_id, group, label 外的列）
    feature_cols_all = [c for c in merged.columns if c not in ["image_id","group","label"]]

    # 处理 NaN / Inf
    for df in [train_df, val_df, test_df]:
        df[feature_cols_all] = df[feature_cols_all].replace([np.inf, -np.inf], np.nan)
        df[feature_cols_all] = df[feature_cols_all].fillna(0.0)

    # 仅在训练集上拟合 scaler，然后应用到所有集合
    scaler = StandardScaler()
    X_train = train_df[feature_cols_all].values.astype(np.float32)
    scaler.fit(X_train)

    train_df[feature_cols_all] = scaler.transform(train_df[feature_cols_all].values.astype(np.float32))
    val_df[feature_cols_all]   = scaler.transform(val_df[feature_cols_all].values.astype(np.float32))
    test_df[feature_cols_all]  = scaler.transform(test_df[feature_cols_all].values.astype(np.float32))

    # 删除低方差特征（在训练集上计算方差）
    var = train_df[feature_cols_all].var(axis=0).values
    good_mask = var > 1e-6
    good_cols = [f for f, keep in zip(feature_cols_all, good_mask) if keep]
    if len(good_cols) == 0:
        raise ValueError("All features have zero variance after scaling.")

    train_df = train_df[['image_id','group','label'] + good_cols].copy()
    val_df   = val_df[['image_id','group','label'] + good_cols].copy()
    test_df  = test_df[['image_id','group','label'] + good_cols].copy()

    print(f"✅ Scaled and selected features: {len(good_cols)} dims. shapes -> Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df


# 7️⃣ 融合 Dataset（显式 dtype 控制）
class FusionDataset(Dataset):
    def __init__(self, df):
        self.features = df.drop(columns=["image_id", "group", "label"]).values.astype(np.float32)
        self.labels = df["label"].values.astype(np.int64)
        self.ids = df["image_id"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label, self.ids[idx]


# 8️⃣ 融合分類器
class FusionClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(x)


# 9️⃣ 评估函数（保持不变）
def evaluate_fusion_model(model, loader, return_details=False):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for feats, labs, ids in loader:
            feats = feats.float().to(DEVICE)   # 显式 float32
            labs = labs.long().to(DEVICE)
            out = model(feats)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labs.cpu().numpy())
            all_ids.extend(ids)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if return_details:
        return all_probs, all_labels, all_ids
    return all_probs[:, 1], all_labels


def find_optimal_threshold(probs, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
    thr_grid = np.linspace(0, 1, 200)
    best_j, best_t = -1, 0.5
    for t in thr_grid:
        preds = (probs > t).astype(int)
        cm = confusion_matrix(labels, preds)
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def calculate_metrics(labels, probs):
    if len(np.unique(labels)) < 2:
        return {
            'AUC': 0.5, 'Accuracy': accuracy_score(labels, (probs > 0.5).astype(int)),
            'Sensitivity': 0.0, 'Specificity': 0.0, 'Precision': 0.0,
            'F1': 0.0, 'MCC': 0.0, 'Threshold': 0.5,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
        }
    thr = find_optimal_threshold(probs, labels)
    preds = (probs > thr).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = recall_score(labels, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0

    return {
        'AUC': roc_auc_score(labels, probs),
        'Accuracy': accuracy_score(labels, preds),
        'Sensitivity': sens,
        'Specificity': spec,
        'Precision': precision_score(labels, preds, zero_division=0),
        'F1': f1_score(labels, preds, zero_division=0),
        'MCC': matthews_corrcoef(labels, preds),
        'Threshold': thr,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }


def print_metrics(m):
    print(f"AUC: {m['AUC']:.4f} | Acc: {m['Accuracy']:.4f}")
    print(f"Sens: {m['Sensitivity']:.4f} | Spec: {m['Specificity']:.4f}")
    print(f"Prec: {m['Precision']:.4f} | F1: {m['F1']:.4f}")
    print(f"MCC: {m['MCC']:.4f} | Th: {m['Threshold']:.4f}")
    print(f"TP: {m['TP']} | TN: {m['TN']} | FP: {m['FP']} | FN: {m['FN']}")


def calc_auc_ci(y_true, y_score, alpha=0.05):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 1.0
    auc = roc_auc_score(y_true, y_score)
    n_pos, n_neg = np.sum(y_true == 1), np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return auc, 0.0, 1.0
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = np.sqrt(
        (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) /
        (n_pos * n_neg)
    )
    z = norm.ppf(1 - alpha / 2)
    return auc, max(0, auc - z * se), min(1, auc + z * se)


def generate_roc_curves_three(train_data, val_data, test_data, save_path='fusion_roc_curves.png'):
    t_labels, t_probs, _ = train_data
    v_labels, v_probs, _ = val_data
    te_labels, te_probs, _ = test_data

    from sklearn.metrics import roc_curve

    def safe_roc(labels, probs):
        if len(np.unique(labels)) < 2:
            return [0, 1], [0, 1], 0.5
        return roc_curve(labels, probs)

    fpr_t, tpr_t, _ = safe_roc(t_labels, t_probs)
    fpr_v, tpr_v, _ = safe_roc(v_labels, v_probs)
    fpr_te, tpr_te, _ = safe_roc(te_labels, te_probs)

    tr_auc, tr_lo, tr_hi = calc_auc_ci(t_labels, t_probs)
    v_auc, v_lo, v_hi = calc_auc_ci(v_labels, v_probs)
    te_auc, te_lo, te_hi = calc_auc_ci(te_labels, te_probs)

    plt.style.use('ggplot')
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    plt.figure(figsize=(9, 7))
    plt.plot(fpr_t, tpr_t, color='red', lw=2,
             label=f'Train AUC={tr_auc:.3f} ({tr_lo:.3f}-{tr_hi:.3f})')
    plt.plot(fpr_v, tpr_v, color='blue', lw=2,
             label=f'Valid AUC={v_auc:.3f} ({v_lo:.3f}-{v_hi:.3f})')
    plt.plot(fpr_te, tpr_te, color='green', lw=2,
             label=f'Test AUC={te_auc:.3f} ({te_lo:.3f}-{te_hi:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, ls='--')
    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Fusion Model)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'ROC curves saved to {save_path}')


# 🔟 主训练流程
def train_fusion_model():
    # 加载预训练 ResNet50
    class MedicalResNet50(nn.Module):
        def __init__(self, dropout: float = 0.5):
            super().__init__()
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                backbone = models.resnet50(weights=weights)
            except:
                backbone = models.resnet50(pretrained=True)
            self.features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4, backbone.avgpool
            )
            in_features = backbone.fc.in_features
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))

        def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))
        def forward_features(self, x): return torch.flatten(self.features(x), 1)

    model = MedicalResNet50(dropout=DROP_OUT).to(DEVICE)
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded pretrained backbone weights from {PRETRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load pretrained model from {PRETRAINED_MODEL_PATH}: {e}")
        # 继续使用未加载权重的模型

    model.eval()

    # 提取并融合特征
    train_df, val_df, test_df = load_and_merge_features(model)

    # 创建 Dataloader
    train_set = FusionDataset(train_df)
    val_set = FusionDataset(val_df)
    test_set = FusionDataset(test_df)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化融合模型
    input_dim = train_set.features.shape[1]
    fusion_model = FusionClassifier(input_dim=input_dim, dropout=DROP_OUT).to(DEVICE)
    fusion_model.float() # 强制所有模型参数为 float32，解决 RuntimeError

    # 损失函数 & 优化器
    # 使用更稳健的类别权重：n/(num_classes * count)
    class_counts = np.bincount(train_df["label"].values, minlength=2)
    n_samples = len(train_df)
    num_classes = 2
    if (class_counts == 0).any():
        # 若某类在训练集中数量为 0，使用均等权重作为兜底
        weights_list = [1.0, 1.0]
        print("Warning: One of the classes has zero samples in training data. Using equal class weights.")
    else:
        weights_list = []
        for i in range(num_classes):
            if class_counts[i] > 0:
                weights_list.append(n_samples / (num_classes * class_counts[i]))
            else:
                weights_list.append(0.0)
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    # 训练循环
    BEST_PATH = "best_fusion_model.pth"
    best_mean_auc = -1.0
    best_val_metrics = None
    epochs_no_improve = 0
    logger = open("fusion_training_log.csv", "w", encoding="utf-8")
    logger.write("epoch,train_loss,train_auc,val_auc,test_auc,mean_auc,threshold,val_thr\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        fusion_model.train()
        batch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False, colour='cyan')
        for feats, labs, _ in pbar:
            feats = feats.float().to(DEVICE)
            labs = labs.long().to(DEVICE)
            optimizer.zero_grad()
            logits = fusion_model(feats)
            loss = criterion(logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(batch_losses):.4f}")

        scheduler.step()

        # 评估
        train_probs, train_labels = evaluate_fusion_model(fusion_model, train_loader)
        val_probs, val_labels = evaluate_fusion_model(fusion_model, val_loader)
        test_probs, test_labels = evaluate_fusion_model(fusion_model, test_loader)

        train_m = calculate_metrics(train_labels, train_probs)
        val_m = calculate_metrics(val_labels, val_probs)
        test_m = calculate_metrics(test_labels, test_probs)
        mean_auc = np.mean([train_m['AUC'], val_m['AUC'], test_m['AUC']])

        logger.write(f"{epoch},{np.mean(batch_losses):.6f},{train_m['AUC']:.6f},{val_m['AUC']:.6f},{test_m['AUC']:.6f},"
                     f"{mean_auc:.6f},{val_m['Threshold']:.5f},{find_optimal_threshold(val_probs, val_labels):.5f}\n")
        logger.flush()

        # 保存条件
        qualified = mean_auc > best_mean_auc
        
        if qualified:
            best_mean_auc = mean_auc
            best_val_metrics = val_m.copy()
            torch.save(fusion_model.state_dict(), BEST_PATH)
            epochs_no_improve = 0
            print("\n>>> NEW BEST MODEL saved based on mean_auc <<<")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs.")
            break

        print(f"\nEpoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.6f} | MeanAUC={mean_auc:.4f}")
        for name, m in zip(['Train', 'Valid', 'Test'], [train_m, val_m, test_m]):
            print(name); print_metrics(m)

    logger.close()

    # 加载最佳模型
    if os.path.exists(BEST_PATH):
        print("\nLoading best model...")
        fusion_model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    fusion_model.eval()

    # 最终预测
    tr_pred = evaluate_fusion_model(fusion_model, train_loader, return_details=True)
    val_pred = evaluate_fusion_model(fusion_model, val_loader, return_details=True)
    te_pred = evaluate_fusion_model(fusion_model, test_loader, return_details=True)

    # 保存预测结果
    results = []
    metrics_list = []
    for pred, name, df in zip([tr_pred, val_pred, te_pred], ['Train', 'Valid', 'Test'],
                              [train_df, val_df, test_df]):
        probs, labels, ids = pred
        # 确保在计算最终指标时，阈值是基于该数据集的最佳阈值
        thr = find_optimal_threshold(probs[:, 1], labels) 
        df_out = pd.DataFrame({
            'group': name.lower(), 'image_id': ids, 'true_label': labels,
            'prob_0': probs[:, 0], 'prob_1': probs[:, 1],
            'pred': (probs[:, 1] > thr).astype(int)
        })
        results.append(df_out)
        m = calculate_metrics(labels, probs[:, 1])
        m_row = pd.DataFrame([{'group': name.lower(), **m}])
        metrics_list.append(m_row)
        print(f"\n[{name}] Final Metrics"); print_metrics(m)

    pd.concat(results, ignore_index=True).to_csv("fusion_all_predictions.csv", index=False)
    pd.concat(metrics_list, ignore_index=True).to_csv("fusion_all_metrics.csv", index=False)
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv("fusion_all_features.csv", index=False)
    print("\nPredictions, metrics, and features saved.")

    # ROC 曲线
    if all(len(p[1]) > 0 for p in [tr_pred, val_pred, te_pred]):
        generate_roc_curves_three(
            (tr_pred[1], tr_pred[0][:, 1], tr_pred[2]),
            (val_pred[1], val_pred[0][:, 1], val_pred[2]),
            (te_pred[1], te_pred[0][:, 1], te_pred[2])
        )

    print("\n=== Dataset Summary ===")
    for name, df in [("Train", train_df), ("Valid", val_df), ("Test", test_df)]:
        print(f"{name}: Class-0={sum(df.label==0)}, Class-1={sum(df.label==1)}")

    return best_val_metrics


# 🚀 主程序
if __name__ == '__main__':
    try:
        best_val = train_fusion_model()
        if best_val:
            print("\nBest validation metrics:")
            print_metrics(best_val)
        else:
            print("\nNo qualified checkpoint was saved.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

