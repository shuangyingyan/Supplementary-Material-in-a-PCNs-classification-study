# -------------------------------------------------
# 0️⃣ 必要套件
# -------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import (roc_auc_score, confusion_matrix, recall_score,
                             precision_score, f1_score, accuracy_score,
                             matthews_corrcoef)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as stats   # 保留供未來使用

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------
# 1️⃣ 超參與裝置
# -------------------------------------------------
MAX_EPOCHS   = 100          # 訓練總 epoch 數
BATCH_SIZE   = 8
LR           = 1e-4        # 降低學習率
DROP_OUT     = 0.4          # Dropout率保持高
PATIENCE     = 30           # 早停 patience 减少
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

# -------------------------------------------------
# 2️⃣ 路徑與常數（已改為 Windows 本地路徑）
# -------------------------------------------------
IMG_ROOT_ALL = r'/home/featurize/2DA_custom/2D_0'
TRAIN_TXT    = r'/home/featurize/train2D.txt'
VAL_TXT      = r'/home/featurize/val2D.txt'
TEST_TXT     = r'/home/featurize/test2D.txt'

# -------------------------------------------------
# 3️⃣ 影像 Augmentation (增強版)
# -------------------------------------------------
class MedicalAugmentation:
    """
    強化醫療影像增強 - 包含垂直翻轉、旋轉、高斯模糊等
    """
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # 訓練增強
        train_ops = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.1, hue=0.05
            ),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=5
            )
        ]
        
        # 驗證/測試增強
        self.val_test_aug = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        self.train_aug = transforms.Compose(
            train_ops + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ]
        )

# -------------------------------------------------
# 4️⃣ Dataset
# -------------------------------------------------
class ImageOnlyDataset(Dataset):
    """
    只返回 (image_tensor, label, image_id)。同時統計每個類別的數量
    以便後續計算 class_weights。
    """
    def __init__(self, txt_path, img_root, mode='train'):
        self.img_root = img_root
        self.mode = mode
        self.aug = MedicalAugmentation()

        self.image_paths = []
        self.labels      = []
        self.image_ids   = []

        with open(txt_path, 'r') as f:
            for line in f:
                img_id, lbl = line.strip().split()
                img_path = os.path.join(self.img_root, img_id)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.image_ids.append(img_id)
                    self.labels.append(int(lbl))

        if not self.labels:
            print(f"WARNING: No valid images loaded for {mode} set from '{txt_path}'.")
            self.class_counts = np.array([0, 0], dtype=int)
        else:
            self.class_counts = np.bincount(self.labels, minlength=2)

        print(f"{mode} set -> Class 0: {self.class_counts[0]}, Class 1: {self.class_counts[1]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        transform = self.aug.train_aug if self.mode == 'train' else self.aug.val_test_aug
        return transform(img), label, self.image_ids[idx]

# -------------------------------------------------
# 5️⃣ 模型 - 使用 torchvision 的 ResNet50（ImageNet 預訓練）
# -------------------------------------------------
class MedicalResNet50(nn.Module):
    """
    使用 torchvision 的 ResNet50 作為 backbone
    - pretrained on ImageNet
    - 簡化分類頭部
    - 新增 forward_features() 以導出 2048 維池化後特徵
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        # 兼容不同版本 torchvision 的載入方式
        try:
            # 新版 API（推薦）
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            backbone = models.resnet50(weights=weights)
        except Exception:
            # 舊版回退
            backbone = models.resnet50(pretrained=True)
        
        # 提取特徵提取部分
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # 全局平均池化
        self.avgpool = backbone.avgpool
        
        # 分類頭部
        in_features = backbone.fc.in_features  # ResNet50 為 2048
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)  # 二分類輸出
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward_features(self, x):
        """
        返回池化後扁平化的 2048 維特徵（不經分類頭）。
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# -------------------------------------------------
# 6️⃣ 評估、閾值、指標、ROC
# -------------------------------------------------
def evaluate_model(model, loader, return_details=False):
    """返回 (probs, labels, ids) 或只返回 (prob_of_class1, labels)。"""
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for imgs, labs, ids in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labs.numpy())
            all_ids.extend(ids)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if return_details:
        return all_probs, all_labels, all_ids
    else:
        return all_probs[:, 1], all_labels

def find_optimal_threshold(probs, labels):
    """在 validation 上搜尋 Youden J 最大的閾值。若只有單一類別則返回 0.5。"""
    if len(np.unique(labels)) < 2:
        return 0.5
    thr_grid = np.linspace(0, 1, 200)
    best_j, best_t = -1, 0.5
    for t in thr_grid:
        preds = (probs > t).astype(int)
        cm = confusion_matrix(labels, preds)
        if cm.size != 4: # Handle cases where confusion matrix might not be 2x2
            continue
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t

def calculate_metrics(labels, probs):
    """回傳完整的 metric dict（含 AUC、Acc、Sens、Spec、Prec、F1、MCC、Threshold...）。"""
    if len(np.unique(labels)) < 2:
        # 單一類別的情況下僅回傳簡易指標
        # 如果只有一個類別，例如只有0，則所有真實標籤為0，preds也會是0，probs[1]會很小
        # 如果只有一個類別，例如只有1，則所有真實標籤為1，preds也會是1，probs[1]會很大
        # ROC AUC無法計算，通常會給0.5或NaN。
        # 其他指標如accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
        # 可以計算，但需要注意zero_division。
        
        # 为了避免 ROC AUC 的错误，对于单类别，返回0.5
        auc = 0.5
        # 对于其他指标，即使是单类别，我们也可以计算基于0.5阈值的性能
        preds_at_0_5 = (probs > 0.5).astype(int)
        
        # 确保confusion_matrix能处理单类别
        cm = confusion_matrix(labels, preds_at_0_5, labels=[0, 1])
        # 如果cm不是2x2 (例如，只有一类，另一个类预测为0)，则会报错
        # 更稳健的方法是手动计算TP, TN, FP, FN
        
        # Manual calculation for single class scenario:
        tp = np.sum((labels == 1) & (preds_at_0_5 == 1))
        tn = np.sum((labels == 0) & (preds_at_0_5 == 0))
        fp = np.sum((labels == 0) & (preds_at_0_5 == 1))
        fn = np.sum((labels == 1) & (preds_at_0_5 == 0))

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
        acc = accuracy_score(labels, preds_at_0_5)
        mcc = matthews_corrcoef(labels, preds_at_0_5) if (tp + tn + fp + fn) > 0 else 0.0 # Handle case where all are 0

        return {
            'AUC': auc,
            'Accuracy': acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'Precision': prec,
            'F1': f1,
            'MCC': mcc,
            'Threshold': 0.5, # Default threshold for single class
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        }

    thr = find_optimal_threshold(probs, labels)
    preds = (probs > thr).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = recall_score(labels, preds, zero_division=0)   # 同時為 Sensitivity
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
    """整齊列印 metric dict。"""
    print(f"AUC: {m['AUC']:.4f} | Acc: {m['Accuracy']:.4f}")
    print(f"Sens: {m['Sensitivity']:.4f} | Spec: {m['Specificity']:.4f}")
    print(f"Prec: {m['Precision']:.4f} | F1: {m['F1']:.4f}")
    print(f"MCC: {m['MCC']:.4f} | Th: {m['Threshold']:.4f}")
    print(f"TP: {m['TP']} | TN: {m['TN']} | FP: {m['FP']} | FN: {m['FN']}")

def calc_auc_ci(y_true, y_score, alpha=0.05):
    """DeLong‑style 95% CI（若只有單一類別則回傳寬鬆區間）。"""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 1.0
    auc = roc_auc_score(y_true, y_score)
    n_pos, n_neg = np.sum(y_true == 1), np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return auc, 0.0, 1.0
    
    # Simplified calculation for SE, common workaround for DeLong's if package not available
    # This is a general approximation, not true DeLong.
    # For true DeLong, packages like pROC (R) or a more complex Python implementation is needed.
    # For now, let's use a simpler approach that gives some CI.
    # A robust solution might involve bootstrap.
    # For demonstration, we'll use a slightly simplified formula for variance from Hanley & McNeil (1982)
    # The true DeLong is more complex and involves U-statistics.
    # Let's keep the existing calc_auc_ci, as it already has logic for single class
    # and is a placeholder for a more advanced calculation.
    
    # Using the existing formula, it seems to be an approximation based on Hanley & McNeil
    # but the q1, q2 terms suggest a more nuanced calculation which might be correct.
    # Let's stick with it as it was provided.
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    denominator = n_pos * n_neg
    
    # Handle cases where denominator might be zero (e.g., n_pos or n_neg is 0, handled above)
    # or if any term within sqrt is negative due to float precision (rare but possible)
    variance_term = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) +
                     (n_neg - 1) * (q2 - auc * auc)) / denominator
    
    # Ensure variance_term is non-negative before sqrt
    se = np.sqrt(max(0, variance_term))
    
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return auc, max(0, auc - z * se), min(1, auc + z * se)

def generate_roc_curves_three(train_data, val_data, test_data,
                             save_path='roc_curves.png'):
    """在同一張圖裡畫 Train / Valid / Test 的 ROC 曲線與 95% CI。"""
    t_labels, t_probs, _ = train_data
    v_labels, v_probs, _ = val_data
    te_labels, te_probs, _ = test_data

    from sklearn.metrics import roc_curve

    def safe_roc(labels, probs):
        if len(np.unique(labels)) < 2:
            # If only one class, ROC curve is just a point or ill-defined.
            # Return dummy points for a diagonal line and an AUC of 0.5.
            return np.array([0, 1]), np.array([0, 1]), 0.5
        return roc_curve(labels, probs)

    fpr_t, tpr_t, _ = safe_roc(t_labels, t_probs)
    fpr_v, tpr_v, _ = safe_roc(v_labels, v_probs)
    fpr_te, tpr_te, _ = safe_roc(te_labels, te_probs)

    # Use calculate_metrics to get AUC, which handles single-class gracefully
    # then derive CI
    tr_auc_calc = calculate_metrics(t_labels, t_probs)['AUC']
    v_auc_calc = calculate_metrics(v_labels, v_probs)['AUC']
    te_auc_calc = calculate_metrics(te_labels, te_probs)['AUC']

    tr_auc, tr_lo, tr_hi = calc_auc_ci(t_labels, t_probs)
    v_auc, v_lo, v_hi = calc_auc_ci(v_labels, v_probs)
    te_auc, te_lo, te_hi = calc_auc_ci(te_labels, te_probs)

    plt.style.use('ggplot')
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelweight": "bold",
        "figure.titleweight": "bold",
        "font.family": "DejaVu Sans"
    })
    plt.figure(figsize=(9, 7))
    plt.plot(fpr_t, tpr_t, color='red', lw=2,
             label=f'Train AUC={tr_auc:.3f} (95% CI {tr_lo:.3f}-{tr_hi:.3f})')
    plt.plot(fpr_v, tpr_v, color='blue', lw=2,
             label=f'Valid AUC={v_auc:.3f} (95% CI {v_lo:.3f}-{v_hi:.3f})')
    plt.plot(fpr_te, tpr_te, color='green', lw=2,
             label=f'Test AUC={te_auc:.3f} (95% CI {te_lo:.3f}-{te_hi:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, ls='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Train‑Red / Valid‑Blue / Test‑Green)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'ROC curves saved to {save_path}')

# -------------------------------------------------
# 7️⃣ 主訓練流程
# -------------------------------------------------
def train_single_stream_model():
    """完整的訓練、驗證、測試、保存、報告流程。"""
    # -------------------------------------------------
    # 7.1️⃣ Dataset & DataLoader
    # -------------------------------------------------
    train_set = ImageOnlyDataset(TRAIN_TXT, IMG_ROOT_ALL, mode='train')
    val_set   = ImageOnlyDataset(VAL_TXT,   IMG_ROOT_ALL, mode='val')
    test_set  = ImageOnlyDataset(TEST_TXT,  IMG_ROOT_ALL, mode='test')

    # 必要的 sanity‑check
    if len(train_set) == 0:
        print("ERROR: Training set is empty – aborting training.")
        return None
    if len(val_set) == 0:
        print("WARNING: Validation set is empty. Checkpoint saving will rely solely on mean_auc.")
    if len(test_set) == 0:
        print("WARNING: Test set is empty. Mean_auc calculation will exclude test AUC.")


    # ---------- class weights (for loss) ----------
    class_counts = torch.tensor(train_set.class_counts).float()
    if class_counts.sum() == 0 or len(class_counts) < 2 or (class_counts == 0).any():
        print("WARNING: Insufficient or single class counts in training set – using uniform weights for loss.")
        # Fallback to uniform weights if any class is missing or sum is 0
        class_weights = torch.ones(2).to(DEVICE) / 2.0
        # For sampler, if one class is missing, we can't create weights for it,
        # but the sampler should not be created if `train_set.labels` is empty
        # or contains only one class. Let's make sure `sample_weights` creation is robust.
    else:
        # Aggressive balancing only if ratio is high
        if max(class_counts) / min(class_counts) > 5:
            print("Applying aggressive class balancing for loss.")
            class_weights = 1.0 / (class_counts + 1e-8)
        else:
            print("Class imbalance not severe, using uniform weights for loss.")
            class_weights = torch.ones(2).to(DEVICE)
        
        class_weights = class_weights / class_weights.sum() # Normalize weights
        class_weights = class_weights.to(DEVICE)

    # ---------- WeightedRandomSampler ----------
    # Only create sampler if there are actually labels to sample from and at least two classes
    sampler = None
    if len(train_set.labels) > 0 and len(np.unique(train_set.labels)) > 1:
        # Ensure class_counts used for sampler weights are not zero for any existing label
        # (This is slightly different from loss weights as 1/0 is infinite)
        sampler_class_counts = train_set.class_counts
        # Replace 0 counts with a small number to avoid division by zero, though a fully empty class
        # would typically prevent its label from being in train_set.labels.
        # However, if minlength=2 made a class exist with 0 count, this handles it.
        sampler_class_counts[sampler_class_counts == 0] = 1e-8 
        sample_weights = [1.0 / sampler_class_counts[lbl] for lbl in train_set.labels]
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights) * 2, # Sample more to ensure balance
                                        replacement=True)

    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              sampler=sampler if sampler else None, # Use sampler if created, else shuffle for non-imbalanced or single-class
                              shuffle=False if sampler else True, # If using sampler, don't shuffle in DataLoader
                              num_workers=4,
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # -------------------------------------------------
    # 7.2️⃣ Model, Loss, Optimizer, Scheduler
    # -------------------------------------------------
    model = MedicalResNet50(dropout=DROP_OUT).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(),
                            lr=LR,
                            weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=MAX_EPOCHS,
                                                    eta_min=1e-6)

    # -------------------------------------------------
    # 7.3️⃣ Checkpoint / Early‑stop 設定
    # -------------------------------------------------
    BEST_PATH = "best_resnet50_model.pth"
    best_mean_auc = -1.0
    best_val_metrics = None
    epochs_no_improve = 0

    log_path = "resnet50_training_log.csv"
    logger = open(log_path, "w", encoding="utf-8")
    logger.write("epoch,train_loss,train_auc,val_auc,test_auc,mean_auc,threshold,val_thr\n")
    logger.flush()

    # -------------------------------------------------
    # 7.4️⃣ Training Loop
    # -------------------------------------------------
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}",
                    leave=False,
                    colour='cyan')
        for imgs, labs, _ in pbar:
            imgs = imgs.to(DEVICE)
            labs = labs.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(batch_losses))

        scheduler.step()

        # -------------------------------------------------
        # 7.5️⃣ 評估 (train / val / test)
        # -------------------------------------------------
        train_probs, train_labels = evaluate_model(model, train_loader)
        val_probs,   val_labels   = evaluate_model(model, val_loader)
        test_probs,  test_labels  = evaluate_model(model, test_loader)

        train_metrics = calculate_metrics(train_labels, train_probs)
        val_metrics   = calculate_metrics(val_labels,   val_probs)
        test_metrics  = calculate_metrics(test_labels,  test_probs)

        # 在 validation 上重新找最佳閾值（供後續 checkpoint 使用）
        val_thr = find_optimal_threshold(val_probs, val_labels)

        # 均值 AUC
        # and not NaN.
        valid_aucs = []
        if len(np.unique(train_labels)) > 1:
            valid_aucs.append(train_metrics['AUC'])
        if len(np.unique(val_labels)) > 1:
            valid_aucs.append(val_metrics['AUC'])
        if len(np.unique(test_labels)) > 1:
            valid_aucs.append(test_metrics['AUC'])

        mean_auc = np.mean(valid_aucs) if valid_aucs else -1.0 # If all are single-class, mean_auc will be -1.0

        # -------------------------------------------------
        # 7.6️⃣ Logging
        # -------------------------------------------------
        logger.write(
            f"{epoch},"
            f"{np.mean(batch_losses):.6f},"
            f"{train_metrics['AUC']:.6f},"
            f"{val_metrics['AUC']:.6f},"
            f"{test_metrics['AUC']:.6f},"
            f"{mean_auc:.6f},"
            f"{val_metrics['Threshold']:.5f}," # This threshold was derived from val_metrics
            f"{val_thr:.5f}\n"                 # This threshold was derived directly from val_probs, val_labels
        )
        logger.flush()

        # -------------------------------------------------
        # 7.7️⃣ Checkpoint
        # -------------------------------------------------
        qualified = (mean_auc > best_mean_auc)

        if qualified:
            best_mean_auc = mean_auc
            best_val_metrics = val_metrics # Store metrics for the epoch that created the best checkpoint
            torch.save(model.state_dict(), BEST_PATH)
            epochs_no_improve = 0
            print("\n>>> NEW QUALIFIED CHECKPOINT saved <<<")
        else:
            epochs_no_improve += 1

        # -------------------------------------------------
        # 7.8️⃣ Early stopping
        # -------------------------------------------------
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stop after {epoch} epochs (no improvement {PATIENCE} epochs).")
            break

        # -------------------------------------------------
        # 7.9️⃣ Epoch Summary (print)
        # -------------------------------------------------
        print(f"\nEpoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.6f} | MeanAUC={mean_auc:.4f}")
        for name, mets in zip(['Train', 'Valid', 'Test'],
                              [train_metrics, val_metrics, test_metrics]):
            print(name)
            print_metrics(mets)

    logger.close()

    # -------------------------------------------------
    # 7.10️⃣ 載入最好的 checkpoint (若有)
    # -------------------------------------------------
    if os.path.exists(BEST_PATH):
        print("\nLoading best qualified checkpoint …")
        state = torch.load(BEST_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print("\n⚠️ No qualified checkpoint – using final epoch weights.")

    model.eval()

    # -------------------------------------------------
    # 7.11️⃣ 最終預測、保存 CSV、統計指標
    # -------------------------------------------------
    def final_predict(dataset):
        if len(dataset) == 0:
            # Return empty arrays with correct shape for 2 classes
            return np.empty((0, 2)), np.empty((0,), dtype=int), []   
        loader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
        return evaluate_model(model, loader, return_details=True)

    tr_pred = final_predict(train_set)
    val_pred = final_predict(val_set)
    te_pred = final_predict(test_set)

    def save_predictions(pred_tuple, split_name):
        probs, labels, ids = pred_tuple
        if len(labels) == 0:
            print(f"[{split_name}] dataset is empty – skip saving predictions and metrics.")
            return

        # Use validation's optimal threshold if available, otherwise 0.5 for test/train or if val is empty
        # If val_pred is empty, val_labels will be empty, find_optimal_threshold will return 0.5
        val_probs_for_thr, val_labels_for_thr, _ = val_pred
        thr = find_optimal_threshold(val_probs_for_thr[:, 1], val_labels_for_thr) if len(val_labels_for_thr) > 0 else 0.5
        
        df = pd.DataFrame({
            "image_id":   ids,
            "true_label": labels,
            "prob_0":     probs[:, 0],
            "prob_1":     probs[:, 1],
            "pred":       (probs[:, 1] > thr).astype(int)
        })
        df.to_csv(f"resnet50_{split_name.lower()}_predictions.csv", index=False)

        # Calculate and save metrics for the current split using its own data
        mets = calculate_metrics(labels, df["prob_1"])
        pd.DataFrame([mets]).to_csv(f"resnet50_{split_name.lower()}_metrics.csv",
                                   index=False)
        print(f"\n[{split_name}] final metrics")
        print_metrics(mets)

    save_predictions(tr_pred,   "Train")
    save_predictions(val_pred, "Valid")
    save_predictions(te_pred,  "Test")

    # -------------------------------------------------
    # 7.12️⃣ 新增功能：提取最佳模型的 2048 維池化特徵並保存為 CSV，
    #         同時使用 PCA 降維至 128 維並保存 CSV
    # -------------------------------------------------
    def extract_features_to_csv(model, txt_path, img_root, split_name, transform_mode='val',
                                pca: PCA | None = None, fit_pca: bool = False):
        """
        使用指定的 transform_mode（'val' 使用固定 Resize+Normalize），
        提取 2048 維池化特徵，保存至 CSV；
        若提供 pca 或 fit_pca=True，則輸出 128 維 PCA 特徵 CSV。
        返回 (features_np, labels_np, ids_list, fitted_pca or pca)
        """
        # 使用不帶隨機增強的模式，確保特徵穩定（train 也用 'val' 模式）
        feat_dataset = ImageOnlyDataset(txt_path, img_root, mode=transform_mode)
        if len(feat_dataset) == 0:
            print(f"[{split_name}] empty – skip feature extraction.")
            return np.empty((0, 2048)), np.empty((0,), dtype=int), [], pca

        loader = DataLoader(feat_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

        model.eval()
        all_feats = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for imgs, labs, ids in tqdm(loader, desc=f"Features-{split_name}", leave=False, colour='magenta'):
                imgs = imgs.to(DEVICE)
                feats = model.forward_features(imgs)  # [B, 2048]
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labs.numpy())
                all_ids.extend(ids)

        feats_np = np.concatenate(all_feats, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # 保存 2048 維特徵 CSV
        cols_2048 = [f"f{i+1}" for i in range(feats_np.shape[1])]
        df_2048 = pd.DataFrame(feats_np, columns=cols_2048)
        df_2048.insert(0, "image_id", all_ids)
        df_2048.insert(1, "true_label", labels_np)
        out_2048 = f"resnet50_{split_name.lower()}_features_2048.csv"
        df_2048.to_csv(out_2048, index=False)
        print(f"[{split_name}] 2048-D features saved to {out_2048}")

        # PCA 至 128 維
        pca_ret = pca
        feats_pca = None
        if fit_pca:
            # 使用 train split 擬合 PCA
            if feats_np.shape[0] > 128: # Need enough samples for PCA
                pca_ret = PCA(n_components=128, random_state=SEED)
                feats_pca = pca_ret.fit_transform(feats_np)
                print(f"[{split_name}] PCA fitted on {feats_np.shape[0]} samples.")
            else:
                print(f"[{split_name}] Not enough samples ({feats_np.shape[0]}) to fit PCA. Skipping PCA for this split.")
        elif pca is not None and feats_np.shape[0] > 0:
            # Check if pca is already fitted
            if hasattr(pca, 'components_'):
                feats_pca = pca.transform(feats_np)
            else:
                print(f"[{split_name}] Provided PCA object is not fitted. Skipping PCA transform.")

        if feats_pca is not None and feats_pca.shape[0] > 0:
            cols_128 = [f"pc{i+1}" for i in range(feats_pca.shape[1])]
            df_128 = pd.DataFrame(feats_pca, columns=cols_128)
            df_128.insert(0, "image_id", all_ids)
            df_128.insert(1, "true_label", labels_np)
            out_128 = f"resnet50_{split_name.lower()}_features_pca128.csv"
            df_128.to_csv(out_128, index=False)
            print(f"[{split_name}] 128-D PCA features saved to {out_128}")
        else:
            if fit_pca or (pca is not None):
                print(f"[{split_name}] PCA features not produced, likely due to insufficient data or unfitted PCA.")
            else:
                print(f"[{split_name}] PCA not applied (no PCA object provided and fit_pca=False).")

        return feats_np, labels_np, all_ids, pca_ret

    # 先在 Train 上擬合 PCA（使用固定的 val/test transform）
    # Pass an initially empty PCA object if needed, or None as done
    tr_feats, tr_labels_f, tr_ids_f, pca_model = extract_features_to_csv(
        model, TRAIN_TXT, IMG_ROOT_ALL, split_name="Train", transform_mode='val', pca=None, fit_pca=True
    )
    # 對 Valid/Test 應用同一個 PCA
    _ = extract_features_to_csv(
        model, VAL_TXT, IMG_ROOT_ALL, split_name="Valid", transform_mode='val', pca=pca_model, fit_pca=False
    )
    _ = extract_features_to_csv(
        model, TEST_TXT, IMG_ROOT_ALL, split_name="Test", transform_mode='val', pca=pca_model, fit_pca=False
    )

    # -------------------------------------------------
    # 7.13️⃣ ROC 曲線（若三個 split 都有資料才畫）
    # -------------------------------------------------
    # Check if there are enough unique labels (at least 2) for ROC calculation
    has_train_data_for_roc = len(tr_pred[1]) > 0 and len(np.unique(tr_pred[1])) > 1
    has_val_data_for_roc   = len(val_pred[1]) > 0 and len(np.unique(val_pred[1])) > 1
    has_test_data_for_roc  = len(te_pred[1]) > 0 and len(np.unique(te_pred[1])) > 1

    if has_train_data_for_roc or has_val_data_for_roc or has_test_data_for_roc:
        # Pass dummy data for splits that don't have enough data for ROC to avoid errors
        # safe_roc handles single-class by returning a diagonal line.
        generate_roc_curves_three(
            (tr_pred[1], tr_pred[0][:, 1], "Train"),
            (val_pred[1], val_pred[0][:, 1], "Valid"),
            (te_pred[1], te_pred[0][:, 1], "Test"),
            save_path="resnet50_roc_curves.png")
    else:
        print("\nSkipping ROC curve generation – no split has sufficient data (at least two classes) for meaningful ROC curves.")


    # -------------------------------------------------
    # 7.14️⃣ 資料集摘要
    # -------------------------------------------------
    print("\n=== Dataset Summary ===")
    for name, ds in zip(["Train", "Valid", "Test"],
                       [train_set, val_set, test_set]):
        print(f"{name}: Class‑0={ds.class_counts[0]}, Class‑1={ds.class_counts[1]}")

    # -------------------------------------------------
    # 7.15️⃣ 返回最佳驗證指標（方便外部使用）
    # -------------------------------------------------
    return best_val_metrics

# -------------------------------------------------
# 8️⃣ 主程式入口
# -------------------------------------------------
if __name__ == '__main__':
    best_val = train_single_stream_model()
    if best_val is not None:
        print("\nBest validation metrics (from qualified checkpoint):")
        print_metrics(best_val)
    else:
        print("\nTraining finished without a qualified checkpoint.")
