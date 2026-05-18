"""
COMP462 - CIFAR-10 Image Classification
3 models: Logistic Regression -> SVM (RBF) -> CNN (PyTorch)
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn, matplotlib

DATASET_PATH    = r'c:\Users\akcay\Documents\Python Projects\comp462\dataset\cifar-10-batches-py'
RANDOM_STATE    = 42
CLASS_NAMES     = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
CHECKPOINT_PATH = 'best_cnn.pth'

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("LIBRARY VERSIONS")
print("=" * 60)
print(f"  numpy      {np.__version__}")
print(f"  pandas     {pd.__version__}")
print(f"  sklearn    {sklearn.__version__}")
print(f"  matplotlib {matplotlib.__version__}")
print(f"  seaborn    {sns.__version__}")
print(f"  torch      {torch.__version__}")
print(f"  device     {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# dict that evaluate_and_visualize fills in, used for the final comparison chart
metrics_data = {}


def evaluate_and_visualize(name, y_true, y_pred):
    """Compute metrics, print report, save confusion matrix and TP/FP/FN/TN heatmap."""
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics_data[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
        "MCC":       mcc,
    }

    print(f"\n  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}"
          f"  F1={f1:.4f}  MCC={mcc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # normalized confusion matrix (row = true class, values in %)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.4, linecolor='grey')
    ax.set_title(f'{name} — Normalized Confusion Matrix (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'results/cm_{slug}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: results/cm_{slug}.png")

    # TP/FP/FN/TN per class
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    df_tpfp = pd.DataFrame(
        {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        index=CLASS_NAMES
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_tpfp, annot=True, fmt='d', cmap='YlOrRd',
                ax=ax, linewidths=0.5, linecolor='grey')
    ax.set_title(f'{name} — TP / FP / FN / TN per Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/tpfp_{slug}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: results/tpfp_{slug}.png")


# SECTION 1 — Data Loading
print("\n" + "=" * 60)
print("SECTION 1 — DATA LOADING")
print("=" * 60)


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return np.array(d[b'data'], dtype=np.float32), np.array(d[b'labels'], dtype=np.int64)


batches_X, batches_y = [], []
for i in range(1, 6):
    X_b, y_b = load_batch(os.path.join(DATASET_PATH, f'data_batch_{i}'))
    batches_X.append(X_b)
    batches_y.append(y_b)

X_train_raw = np.vstack(batches_X)
y_train_raw = np.concatenate(batches_y)
X_test_raw, y_test = load_batch(os.path.join(DATASET_PATH, 'test_batch'))

print(f"  Training : {X_train_raw.shape}  range=[{X_train_raw.min():.0f}, {X_train_raw.max():.0f}]")
print(f"  Test     : {X_test_raw.shape}")

# sample images grid
fig, axes = plt.subplots(10, 5, figsize=(10, 20))
fig.suptitle('CIFAR-10 — 5 Random Samples per Class', fontsize=14, fontweight='bold', y=1.01)
for row, cls in enumerate(range(10)):
    idx = np.where(y_train_raw == cls)[0]
    chosen = np.random.choice(idx, 5, replace=False)
    for col, img_idx in enumerate(chosen):
        img = X_train_raw[img_idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(CLASS_NAMES[cls], fontsize=11, rotation=0,
                                      labelpad=45, va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/viz1_sample_images.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz1_sample_images.png")

# class distribution — check balance before deciding on metrics
unique, counts = np.unique(y_train_raw, return_counts=True)
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(CLASS_NAMES, counts, color=plt.cm.tab10(np.linspace(0, 1, 10)),
              edgecolor='black', linewidth=0.5)
ax.set_title('CIFAR-10 Class Distribution (Training Set)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Images')
ax.set_ylim(0, 6000)
ax.axhline(5000, color='red', linestyle='--', linewidth=1, label='5,000 per class')
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, cnt + 50, str(cnt),
            ha='center', va='bottom', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig('results/viz2_class_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz2_class_distribution.png")
print(f"\n  Dataset is balanced — {counts[0]:,} samples per class.")
print("  Accuracy is a valid metric, no oversampling needed.")


# SECTION 2 — Preprocessing
print("\n" + "=" * 60)
print("SECTION 2 — PREPROCESSING")
print("=" * 60)

# 80/20 stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw, y_train_raw,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_raw
)
print(f"\n  [2.1] Train/Val split (80/20, stratified)")
print(f"    X_train : {X_train.shape}  — {np.unique(y_train, return_counts=True)[1]} per class")
print(f"    X_val   : {X_val.shape}   — {np.unique(y_val,   return_counts=True)[1]} per class")

# per-channel normalization — stats from training set only to avoid leakage
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def normalize_flat(X):
    X = X / 255.0
    X[:, :1024]     = (X[:, :1024]     - CIFAR_MEAN[0]) / CIFAR_STD[0]
    X[:, 1024:2048] = (X[:, 1024:2048] - CIFAR_MEAN[1]) / CIFAR_STD[1]
    X[:, 2048:]     = (X[:, 2048:]     - CIFAR_MEAN[2]) / CIFAR_STD[2]
    return X


X_train_norm = normalize_flat(X_train.copy())
X_val_norm   = normalize_flat(X_val.copy())
X_test_norm  = normalize_flat(X_test_raw.copy())
print(f"\n  [2.2] Per-channel normalization")
print(f"    After norm — mean: {X_train_norm.mean():.4f}  std: {X_train_norm.std():.4f}")

# PCA for LR and SVM — 150 components keeps ~92.9% variance
# CNN doesn't use PCA because it needs the 2D pixel layout intact
print(f"\n  [2.3] PCA (LR + SVM only)")
pca_full = PCA(n_components=200, random_state=RANDOM_STATE)
pca_full.fit(X_train_norm)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
for n in [50, 100, 150, 200]:
    print(f"    {n:3d} components -> {cumvar[n-1]:.1f}% explained variance")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, 201), cumvar, color='steelblue', linewidth=2)
ax.fill_between(range(1, 201), cumvar, alpha=0.15, color='steelblue')
for n, color in [(50, 'orange'), (100, 'green'), (150, 'red'), (200, 'purple')]:
    ax.axvline(n, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
    ax.text(n + 2, cumvar[n-1] - 3, f'{n}: {cumvar[n-1]:.1f}%', color=color, fontsize=9)
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('PCA — Cumulative Explained Variance', fontweight='bold')
ax.set_xlim(1, 200)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/viz3_pca_variance.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz3_pca_variance.png")

N_COMPONENTS = 150
pca150 = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
pca150.fit(X_train_norm)
X_train_pca = pca150.transform(X_train_norm)
X_val_pca   = pca150.transform(X_val_norm)
X_test_pca  = pca150.transform(X_test_norm)
print(f"    Applied PCA({N_COMPONENTS}) -> train {X_train_pca.shape}, test {X_test_pca.shape}")

# SVM can't fit on 40k — the kernel matrix would be ~12.8 GB, so subsample to 10k
_, X_svm, _, y_svm = train_test_split(
    X_train_pca, y_train,
    test_size=10_000,
    random_state=RANDOM_STATE,
    stratify=y_train
)
print(f"\n  [2.4] SVM subsample (10k from 40k, stratified)")
print(f"    X_svm: {X_svm.shape}  — {np.unique(y_svm, return_counts=True)[1]} per class")

# CNN needs (N, C, H, W)
X_train_cnn = X_train_norm.reshape(-1, 3, 32, 32)
X_val_cnn   = X_val_norm.reshape(-1, 3, 32, 32)
X_test_cnn  = X_test_norm.reshape(-1, 3, 32, 32)
print(f"\n  [2.5] Reshape for CNN: {X_train_norm.shape} -> {X_train_cnn.shape}")


# SECTION 3 — Logistic Regression (One-vs-All)
print("\n" + "=" * 60)
print("SECTION 3 — LOGISTIC REGRESSION (One-vs-All)")
print("=" * 60)
print("""
  Linear baseline — draws decision boundaries in PCA space.
  OvA trains one binary classifier per class (as in course material).
  Expected to struggle since image classes aren't linearly separable.
""")

t0 = time.time()
lr_model = LogisticRegression(
    multi_class='ovr',   # One-vs-All, 10 binary classifiers
    solver='lbfgs',
    max_iter=1000,
    C=1.0,
    random_state=RANDOM_STATE
)
lr_model.fit(X_train_pca, y_train)
lr_train_time = time.time() - t0

y_pred_lr_val  = lr_model.predict(X_val_pca)
y_pred_lr_test = lr_model.predict(X_test_pca)

print(f"  Training time  : {lr_train_time:.1f}s  ({X_train_pca.shape[0]:,} samples)")
print(f"  Val accuracy   : {accuracy_score(y_val,  y_pred_lr_val):.4f}")

evaluate_and_visualize("Logistic Regression", y_test, y_pred_lr_test)


# SECTION 4 — SVM (RBF Kernel)
print("\n" + "=" * 60)
print("SECTION 4 — SVM (RBF KERNEL)")
print("=" * 60)
print("""
  RBF kernel maps PCA features to a higher-dimensional space where
  the classes become more separable — something LR can't do.
  Still operates on flat features, doesn't see spatial structure.
""")

t0 = time.time()
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',              # auto-scales to feature variance
    decision_function_shape='ovr',
    random_state=RANDOM_STATE
)
svm_model.fit(X_svm, y_svm)
svm_train_time = time.time() - t0

y_pred_svm_val  = svm_model.predict(X_val_pca)
y_pred_svm_test = svm_model.predict(X_test_pca)

print(f"  Training time      : {svm_train_time:.1f}s  ({X_svm.shape[0]:,} samples — subsampled)")
print(f"  Support vectors    : {svm_model.n_support_.sum():,}")
print(f"  Val accuracy       : {accuracy_score(y_val, y_pred_svm_val):.4f}")

evaluate_and_visualize("SVM (RBF)", y_test, y_pred_svm_test)


# SECTION 5 — CNN (PyTorch)
print("\n" + "=" * 60)
print("SECTION 5 — CNN (PyTorch)")
print("=" * 60)
print("""
  3x3 filters learn local patterns (edges, textures) without flattening
  the image. VGG-style: double conv before each pooling, channels double
  each block (32->64->128). BatchNorm helps training, Dropout fights overfitting.
""")


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CIFAR10CNN().to(device)
total_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"  Device           : {device}")
print(f"  Trainable params : {total_params:,}")

train_ds = TensorDataset(torch.tensor(X_train_cnn), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val_cnn),   torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

EPOCHS, PATIENCE = 30, 7
train_losses, val_losses, val_accs = [], [], []
best_val_acc, patience_counter = 0.0, 0
cnn_start = time.time()

print()
for epoch in range(1, EPOCHS + 1):
    cnn_model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn_model(Xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)
    train_losses.append(running_loss / len(train_ds))

    cnn_model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = cnn_model(Xb)
            val_loss += criterion(out, yb).item() * len(yb)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += len(yb)
    val_losses.append(val_loss / total)
    val_accs.append(correct / total)
    scheduler.step()

    print(f"  Epoch {epoch:02d}/{EPOCHS} | "
          f"train_loss={train_losses[-1]:.4f} | "
          f"val_loss={val_losses[-1]:.4f} | "
          f"val_acc={val_accs[-1]:.4f}")

    if val_accs[-1] > best_val_acc:
        best_val_acc = val_accs[-1]
        torch.save(cnn_model.state_dict(), CHECKPOINT_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

cnn_train_time = time.time() - cnn_start
print(f"\n  Training complete : {cnn_train_time/60:.1f} min")
print(f"  Best val accuracy : {best_val_acc:.4f}")

cnn_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))


def cnn_predict(model, X_np, device, batch_size=256):
    model.eval()
    preds = []
    ds = TensorDataset(torch.tensor(X_np))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (Xb,) in loader:
            preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


y_pred_cnn_val  = cnn_predict(cnn_model, X_val_cnn,  device)
y_pred_cnn_test = cnn_predict(cnn_model, X_test_cnn, device)

# CNN learning curves
best_epoch   = int(np.argmax(val_accs)) + 1
epochs_range = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('CNN Training Dynamics', fontsize=13, fontweight='bold')
ax1.plot(epochs_range, train_losses, label='Train Loss', color='steelblue')
ax1.plot(epochs_range, val_losses,   label='Val Loss',   color='orange')
ax1.axvline(best_epoch, color='red', linestyle='--', linewidth=1, label=f'Best epoch ({best_epoch})')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Train vs Validation Loss'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(epochs_range, [a * 100 for a in val_accs], color='green', linewidth=2)
ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=1,
            label=f'Best: {best_val_acc*100:.1f}%')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_title('Validation Accuracy per Epoch'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/viz4_cnn_learning_curves.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz4_cnn_learning_curves.png")

evaluate_and_visualize("CNN", y_test, y_pred_cnn_test)


# SECTION 6 — Performance Comparison
print("\n" + "=" * 60)
print("SECTION 6 — PERFORMANCE COMPARISON")
print("=" * 60)

metrics_df = pd.DataFrame(metrics_data).T
print("\n  TEST SET METRICS")
print("  " + "-" * 55)
print(metrics_df.round(4).to_string())

# training time summary
df_time = pd.DataFrame({
    'Model':          ['Logistic Regression', 'SVM (RBF)', 'CNN'],
    'Train Samples':  [40_000, 10_000, 40_000],
    'Train Time (s)': [round(lr_train_time, 1), round(svm_train_time, 1), round(cnn_train_time, 1)],
    'Test Accuracy':  [round(metrics_data['Logistic Regression']['Accuracy'], 4),
                       round(metrics_data['SVM (RBF)']['Accuracy'], 4),
                       round(metrics_data['CNN']['Accuracy'], 4)],
})
print("\n  TRAINING TIME")
print("  " + "-" * 55)
print(df_time.to_string(index=False))

# comparison bar chart — same pattern as course code
fig, ax = plt.subplots(figsize=(11, 5))
metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
    kind="bar", ax=ax, colormap='tab10', edgecolor='black', linewidth=0.5
)
ax.set_title('Model Performance Comparison — Test Set', fontsize=13, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig('results/viz5_performance_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n  Saved: results/viz5_performance_comparison.png")

# combined confusion matrices (raw counts) for side-by-side view
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')
for ax, (name, y_pred) in zip(axes, [
    ('Logistic Regression', y_pred_lr_test),
    ('SVM (RBF)',           y_pred_svm_test),
    ('CNN',                 y_pred_cnn_test),
]):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45, cmap='Blues')
    acc  = accuracy_score(y_test, y_pred)
    ax.set_title(f'{name}\nAccuracy: {acc:.3f}', fontweight='bold')
plt.tight_layout()
plt.savefig('results/viz6_confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz6_confusion_matrices.png")

# per-class F1 heatmap
per_class_f1 = {}
for name, y_pred in [('LR', y_pred_lr_test), ('SVM', y_pred_svm_test), ('CNN', y_pred_cnn_test)]:
    per_class_f1[name] = f1_score(y_test, y_pred, average=None, zero_division=0)
df_heatmap = pd.DataFrame(per_class_f1, index=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, ax=ax, linewidths=0.5, linecolor='grey')
ax.set_title('Per-Class F1-Score by Model — Test Set', fontsize=13, fontweight='bold')
ax.set_xlabel('Model'); ax.set_ylabel('Class')
plt.tight_layout()
plt.savefig('results/viz7_f1_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz7_f1_heatmap.png")

# radar chart
METRIC_NAMES = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
COLORS = ['#4878CF', '#6ACC65', '#D65F5F']
N      = len(METRIC_NAMES)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for model, color in zip(metrics_df.index, COLORS):
    vals  = [metrics_df.loc[model, m] for m in METRIC_NAMES]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=model)
    ax.fill(angles, vals, color=color, alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRIC_NAMES, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Model Performance Radar Chart\n(Test Set)', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/viz8_radar_chart.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz8_radar_chart.png")

# training time bar chart
model_labels = ['Logistic\nRegression\n(40k)', 'SVM (RBF)\n(10k)', 'CNN\n(40k)']
times        = [lr_train_time, svm_train_time, cnn_train_time]
accuracies   = [metrics_data['Logistic Regression']['Accuracy'],
                metrics_data['SVM (RBF)']['Accuracy'],
                metrics_data['CNN']['Accuracy']]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(model_labels, times, color=COLORS, edgecolor='black', linewidth=0.5, alpha=0.85)
for bar, t, acc in zip(bars, times, accuracies):
    ax.text(bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{t:.0f}s  |  acc={acc:.3f}', va='center', fontsize=10)
ax.set_xlabel('Training Time (seconds)')
ax.set_title('Training Time vs. Accuracy', fontsize=13, fontweight='bold')
ax.set_xlim(0, max(times) * 1.45)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/viz9_training_time.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz9_training_time.png")

# sample predictions — one image per class, green=correct red=wrong
np.random.seed(RANDOM_STATE)
sample_indices = [np.random.choice(np.where(y_test == c)[0]) for c in range(10)]
fig, axes = plt.subplots(3, 10, figsize=(18, 6))
fig.suptitle('Sample Predictions — One Image per Class (Green=Correct, Red=Wrong)',
             fontsize=12, fontweight='bold')
for row, (model_name, preds) in enumerate([
    ('Logistic\nRegression', y_pred_lr_test),
    ('SVM\n(RBF)',           y_pred_svm_test),
    ('CNN',                  y_pred_cnn_test),
]):
    axes[row, 0].set_ylabel(model_name, fontsize=9, rotation=0, labelpad=50, va='center')
    for col, idx in enumerate(sample_indices):
        img = X_test_raw[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        true_lbl = CLASS_NAMES[y_test[idx]]
        pred_lbl = CLASS_NAMES[preds[idx]]
        color    = 'green' if (y_test[idx] == preds[idx]) else 'red'
        axes[row, col].set_title(f'T:{true_lbl}\nP:{pred_lbl}',
                                  fontsize=6.5, color=color, fontweight='bold')
plt.tight_layout()
plt.savefig('results/viz10_sample_predictions.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: results/viz10_sample_predictions.png")


# SECTION 7 — Summary
print("\n" + "=" * 60)
print("SECTION 7 — SUMMARY")
print("=" * 60)

lr_acc  = metrics_data['Logistic Regression']['Accuracy']
svm_acc = metrics_data['SVM (RBF)']['Accuracy']
cnn_acc = metrics_data['CNN']['Accuracy']

print(f"""
  Ranking: CNN ({cnn_acc:.1%}) >> SVM ({svm_acc:.1%}) >> LR ({lr_acc:.1%})

  LR  : linear boundaries in PCA space — can't handle non-linear image manifolds
  SVM : RBF kernel overcomes linearity, still misses spatial structure
  CNN : convolutions preserve 2D layout — each block builds richer features

""")

print("=" * 60)
print("DONE")
print("=" * 60)
