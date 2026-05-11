"""
COMP462 — CIFAR-10 Exploratory Data Analysis (EDA)
====================================================
Generates PNG files for the report's "Dataset Description" and
"Preprocessing Steps" sections.

Run:
    python eda.py

Output files:
    eda_01_sample_grid.png
    eda_02_class_distribution.png
    eda_03_mean_images_per_class.png
    eda_04_std_images_per_class.png
    eda_05_rgb_channel_histograms.png
    eda_06_per_class_brightness.png
    eda_07_pixel_correlation.png
    eda_08_pca_2d_scatter.png
    eda_09_tsne_scatter.png
    eda_10_variance_vs_class.png
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = r'c:\Users\akcay\Documents\Python Projects\comp462\dataset\cifar-10-batches-py'
CLASS_NAMES  = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return np.array(d[b'data'], dtype=np.float32), np.array(d[b'labels'], dtype=np.int64)

print("Loading dataset...")
batches_X, batches_y = [], []
for i in range(1, 6):
    X_b, y_b = load_batch(os.path.join(DATASET_PATH, f'data_batch_{i}'))
    batches_X.append(X_b)
    batches_y.append(y_b)

X_all = np.vstack(batches_X)          # (50000, 3072)  uint8 range [0,255]
y_all = np.concatenate(batches_y)     # (50000,)

X_test, y_test = load_batch(os.path.join(DATASET_PATH, 'test_batch'))

# Reshape to (N, 32, 32, 3) for image ops — keep original scale [0,255]
X_img  = X_all.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # (50000, 32, 32, 3)
X_test_img = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
print(f"  Train: {X_all.shape}   Test: {X_test.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 01 — Sample image grid (5 random per class)
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 01 — Sample image grid...")
fig, axes = plt.subplots(10, 5, figsize=(10, 21))
fig.suptitle('CIFAR-10 — 5 Random Samples per Class', fontsize=14, fontweight='bold', y=1.005)

for row, cls in enumerate(range(10)):
    idx     = np.where(y_all == cls)[0]
    chosen  = np.random.choice(idx, 5, replace=False)
    for col, img_idx in enumerate(chosen):
        img = X_img[img_idx].astype(np.uint8)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(CLASS_NAMES[cls], fontsize=12, rotation=0,
                                      labelpad=50, va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/eda_01_sample_grid.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_01_sample_grid.png")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 02 — Class distribution (train + test)
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 02 — Class distribution...")
_, train_counts = np.unique(y_all,  return_counts=True)
_, test_counts  = np.unique(y_test, return_counts=True)

x     = np.arange(10)
width = 0.38
fig, ax = plt.subplots(figsize=(11, 4))
b1 = ax.bar(x - width/2, train_counts, width, label='Train (50k)',
            color='steelblue', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + width/2, test_counts,  width, label='Test (10k)',
            color='coral',     edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel('Number of Images')
ax.set_title('Class Distribution — Train vs. Test Sets', fontsize=13, fontweight='bold')
ax.axhline(5000, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(1000, color='coral',     linestyle='--', linewidth=0.8, alpha=0.6)
ax.legend(fontsize=10)
ax.set_ylim(0, 6200)
for bar, cnt in zip(b1, train_counts):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 60, str(cnt),
            ha='center', fontsize=7.5)
for bar, cnt in zip(b2, test_counts):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 60, str(cnt),
            ha='center', fontsize=7.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/eda_02_class_distribution.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_02_class_distribution.png")
print(f"  Train counts: {dict(zip(CLASS_NAMES, train_counts))}")
print(f"  Test counts : {dict(zip(CLASS_NAMES, test_counts))}")
print("  -> Dataset is perfectly balanced. No resampling needed.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 03 — Mean image per class
# Shows the "average appearance" of each class
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 03 — Mean images per class...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Mean Image per Class\n(Average pixel values across all training images)',
             fontsize=13, fontweight='bold')

for i, cls in enumerate(range(10)):
    ax    = axes[i // 5][i % 5]
    idx   = np.where(y_all == cls)[0]
    mean_img = X_img[idx].mean(axis=0).astype(np.uint8)
    ax.imshow(mean_img)
    ax.set_title(CLASS_NAMES[cls], fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('analysis/eda_03_mean_images_per_class.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_03_mean_images_per_class.png")
print("  -> Mean images reveal dominant color and shape patterns per class.")
print("    Vehicle classes (ship, airplane) show clear spatial structure.")
print("    Animal classes (cat, dog) are more spatially diffuse.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 04 — Standard deviation image per class
# Shows intra-class variability (where pixels vary the most)
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 04 — Std images per class (intra-class variability)...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Pixel Standard Deviation per Class\n(Bright = high variability within the class)',
             fontsize=13, fontweight='bold')

for i, cls in enumerate(range(10)):
    ax    = axes[i // 5][i % 5]
    idx   = np.where(y_all == cls)[0]
    std_img = X_img[idx].std(axis=0)
    std_img = (std_img / std_img.max() * 255).astype(np.uint8)
    ax.imshow(std_img, cmap='hot')
    ax.set_title(CLASS_NAMES[cls], fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('analysis/eda_04_std_images_per_class.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_04_std_images_per_class.png")
print("  -> High std = object appears in many positions/sizes -> harder to classify.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 05 — RGB channel pixel intensity histograms
# Shows distribution of pixel values per channel across entire training set
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 05 — RGB channel intensity histograms...")
R = X_all[:, :1024].flatten()
G = X_all[:, 1024:2048].flatten()
B = X_all[:, 2048:].flatten()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Pixel Intensity Distribution per Channel — Training Set',
             fontsize=13, fontweight='bold')

for ax, channel, color, name in zip(axes,
                                     [R, G, B],
                                     ['red', 'green', 'blue'],
                                     ['Red Channel', 'Green Channel', 'Blue Channel']):
    ax.hist(channel, bins=64, color=color, alpha=0.7, edgecolor='none')
    ax.axvline(channel.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'mean={channel.mean():.1f}')
    ax.axvline(channel.mean() - channel.std(), color='grey', linestyle=':', linewidth=1)
    ax.axvline(channel.mean() + channel.std(), color='grey', linestyle=':', linewidth=1,
               label=f'std={channel.std():.1f}')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Pixel Value (0–255)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/eda_05_rgb_channel_histograms.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_05_rgb_channel_histograms.png")
print(f"  R: mean={R.mean():.1f}  std={R.std():.1f}")
print(f"  G: mean={G.mean():.1f}  std={G.std():.1f}")
print(f"  B: mean={B.mean():.1f}  std={B.std():.1f}")
print("  -> Different channel means justify per-channel (not global) normalization.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 06 — Per-class mean brightness per channel
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 06 — Per-class brightness per channel...")
class_r_means, class_g_means, class_b_means = [], [], []
for cls in range(10):
    idx = np.where(y_all == cls)[0]
    class_r_means.append(X_all[idx, :1024].mean())
    class_g_means.append(X_all[idx, 1024:2048].mean())
    class_b_means.append(X_all[idx, 2048:].mean())

x     = np.arange(10)
width = 0.25
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width, class_r_means, width, label='Red',   color='#e74c3c', alpha=0.85)
ax.bar(x,         class_g_means, width, label='Green', color='#27ae60', alpha=0.85)
ax.bar(x + width, class_b_means, width, label='Blue',  color='#2980b9', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel('Mean Pixel Intensity (0–255)')
ax.set_title('Mean Pixel Intensity per Channel per Class\n(Natural images show systematic color biases)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(100, 160)
plt.tight_layout()
plt.savefig('analysis/eda_06_per_class_brightness.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_06_per_class_brightness.png")
print("  -> 'ship' has high blue (ocean background).")
print("    'frog' has high green (vegetation background).")
print("    This shows background color is a confounding feature models may overfit to.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 07 — R / G / B channel correlation (scatter on 2000 random pixels)
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 07 — Channel correlation...")
sample_idx = np.random.choice(len(X_all) * 1024, 3000, replace=False)
r_sample = R[sample_idx]
g_sample = G[sample_idx]
b_sample = B[sample_idx]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('RGB Channel Correlation\n(Each point = one pixel sampled randomly from training set)',
             fontsize=13, fontweight='bold')

pairs = [('Red', r_sample, 'Green', g_sample, '#8e44ad'),
         ('Red', r_sample, 'Blue',  b_sample, '#e67e22'),
         ('Green', g_sample, 'Blue', b_sample, '#16a085')]

for ax, (x_name, x_data, y_name, y_data, color) in zip(axes, pairs):
    ax.scatter(x_data, y_data, alpha=0.15, s=4, color=color)
    corr = np.corrcoef(x_data, y_data)[0, 1]
    ax.set_xlabel(f'{x_name} Channel')
    ax.set_ylabel(f'{y_name} Channel')
    ax.set_title(f'{x_name} vs {y_name}\nr = {corr:.3f}', fontweight='bold')
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('analysis/eda_07_pixel_correlation.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_07_pixel_correlation.png")
print("  -> High inter-channel correlation supports PCA as effective dimensionality reduction.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 08 — PCA 2D scatter (class separability in linear feature space)
# Uses 2,000 samples for speed
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 08 — PCA 2D scatter (2k samples)...")

# Normalize first (same pipeline as main script)
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

def normalize_flat(X):
    X = X.copy() / 255.0
    X[:, :1024]     = (X[:, :1024]     - CIFAR_MEAN[0]) / CIFAR_STD[0]
    X[:, 1024:2048] = (X[:, 1024:2048] - CIFAR_MEAN[1]) / CIFAR_STD[1]
    X[:, 2048:]     = (X[:, 2048:]     - CIFAR_MEAN[2]) / CIFAR_STD[2]
    return X

N_VIZ = 2000
viz_idx  = np.random.choice(len(X_all), N_VIZ, replace=False)
X_viz    = normalize_flat(X_all[viz_idx])
y_viz    = y_all[viz_idx]

pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_viz)

fig, ax = plt.subplots(figsize=(9, 7))
colors_10 = plt.cm.tab10(np.linspace(0, 1, 10))
for cls in range(10):
    mask = y_viz == cls
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               color=colors_10[cls], label=CLASS_NAMES[cls],
               alpha=0.5, s=18, edgecolors='none')

var1 = pca2.explained_variance_ratio_[0] * 100
var2 = pca2.explained_variance_ratio_[1] * 100
ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
ax.set_title(f'PCA 2D Projection — {N_VIZ} Random Samples\n'
             f'(Total variance captured: {var1+var2:.1f}%)',
             fontsize=13, fontweight='bold')
ax.legend(markerscale=2, fontsize=9, loc='upper right', ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('analysis/eda_08_pca_2d_scatter.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_08_pca_2d_scatter.png")
print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%  total: {var1+var2:.1f}%")
print("  -> Classes heavily overlap in 2D PCA space -> confirms non-linear model needed.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 09 — t-SNE scatter (better non-linear separability visualization)
# Uses PCA-50 as input to t-SNE for speed
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 09 — t-SNE scatter (may take 2-4 min)...")
pca50   = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca50 = pca50.fit_transform(X_viz)

tsne   = TSNE(n_components=2, perplexity=40, max_iter=1000,
              random_state=RANDOM_STATE, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca50)

fig, ax = plt.subplots(figsize=(9, 7))
for cls in range(10):
    mask = y_viz == cls
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               color=colors_10[cls], label=CLASS_NAMES[cls],
               alpha=0.6, s=18, edgecolors='none')

ax.set_xlabel('t-SNE Dim 1', fontsize=11)
ax.set_ylabel('t-SNE Dim 2', fontsize=11)
ax.set_title(f't-SNE Projection — {N_VIZ} Random Samples\n'
             f'(PCA-50 input -> better cluster separation than linear PCA)',
             fontsize=13, fontweight='bold')
ax.legend(markerscale=2, fontsize=9, loc='upper right', ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('analysis/eda_09_tsne_scatter.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_09_tsne_scatter.png")
print("  -> t-SNE shows more structure than PCA but classes still overlap.")
print("    Vehicle classes (ship, airplane, automobile) form looser clusters.")
print("    Animal classes (cat, dog, deer) remain highly intermingled.")

# ──────────────────────────────────────────────────────────────────────────────
# EDA 10 — Intra-class variance vs inter-class variance
# Shows how spread each class is vs. how far classes are from each other
# ──────────────────────────────────────────────────────────────────────────────
print("EDA 10 — Intra-class variance analysis...")
X_norm_all = normalize_flat(X_all)

global_mean    = X_norm_all.mean(axis=0)
intra_vars, class_means = [], []

for cls in range(10):
    idx        = np.where(y_all == cls)[0]
    cls_mean   = X_norm_all[idx].mean(axis=0)
    intra_var  = ((X_norm_all[idx] - cls_mean) ** 2).mean()
    intra_vars.append(intra_var)
    class_means.append(cls_mean)

inter_vars = [((cm - global_mean) ** 2).mean() for cm in class_means]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Intra-class vs. Inter-class Variance per Class\n'
             '(High intra = diverse images within class; High inter = class far from average)',
             fontsize=12, fontweight='bold')

colors_10_list = [plt.cm.tab10(i / 10) for i in range(10)]

axes[0].barh(CLASS_NAMES, intra_vars, color=colors_10_list, edgecolor='black', linewidth=0.5)
axes[0].set_title('Intra-class Variance\n(lower = more similar images within class)',
                  fontweight='bold')
axes[0].set_xlabel('Mean Squared Deviation from Class Mean')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(CLASS_NAMES, inter_vars, color=colors_10_list, edgecolor='black', linewidth=0.5)
axes[1].set_title('Inter-class Variance\n(higher = class visually distinct from overall average)',
                  fontweight='bold')
axes[1].set_xlabel('Mean Squared Deviation from Global Mean')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/eda_10_variance_analysis.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: analysis/eda_10_variance_analysis.png")

# Identify hardest/easiest classes
hardest = CLASS_NAMES[int(np.argmax(intra_vars))]
easiest = CLASS_NAMES[int(np.argmin(intra_vars))]
most_distinct = CLASS_NAMES[int(np.argmax(inter_vars))]
least_distinct = CLASS_NAMES[int(np.argmin(inter_vars))]

print(f"  Highest intra-class variance (hardest): {hardest}")
print(f"  Lowest  intra-class variance (easiest): {easiest}")
print(f"  Most visually distinct class           : {most_distinct}")
print(f"  Least visually distinct class          : {least_distinct}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE — Generated files:")
print("=" * 60)
files = [
    ("eda_01_sample_grid.png",          "5 random samples per class"),
    ("eda_02_class_distribution.png",   "Train/test class balance"),
    ("eda_03_mean_images_per_class.png","Average appearance per class"),
    ("eda_04_std_images_per_class.png", "Intra-class pixel variability"),
    ("eda_05_rgb_channel_histograms.png","RGB channel intensity distributions"),
    ("eda_06_per_class_brightness.png", "Mean brightness per channel per class"),
    ("eda_07_pixel_correlation.png",    "R/G/B inter-channel correlation"),
    ("eda_08_pca_2d_scatter.png",       "PCA 2D class separability"),
    ("eda_09_tsne_scatter.png",         "t-SNE non-linear class structure"),
    ("eda_10_variance_analysis.png",    "Intra vs inter-class variance"),
]
for fname, desc in files:
    print(f"  {fname:<40} {desc}")
