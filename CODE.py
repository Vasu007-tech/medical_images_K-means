import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.ndimage import binary_opening, binary_closing

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==============================================================================
# USER CONFIGURATION (Edit these preferences anytime during your lab/viva)
# ==============================================================================
IMAGE_PATH = ""          # Leave "" to pick/upload an image interactively, or paste path e.g. "chest_xray.png"
IMAGE_SIZE = (256, 256)  # Standardized grid size (H, W)
K_LIST = [2, 3, 4, 5]    # Cluster counts to evaluate mathematically
DEFAULT_K = 4            # Value of K displayed in Figure 1 (4-panel pipeline)
ALPHA = 0.15             # Spatial weight (set 0.0 for standard 1D intensity K-Means)


# ==============================================================================
# STAGE 1 & 2: CUSTOM IMAGE ACQUISITION & PREPROCESSING
# ==============================================================================
def get_image_path(preset_path=""):
    """Selects user image via preset path, Colab upload, desktop file dialog, or terminal input."""
    if preset_path and os.path.isfile(preset_path):
        return preset_path

    # 1. Google Colab environment -> trigger browser upload widget
    if "google.colab" in sys.modules:
        from google.colab import files
        print("Upload your X-ray or MRI image (PNG, JPG, BMP, TIF):")
        uploaded = files.upload()
        if not uploaded:
            raise FileNotFoundError("No file uploaded.")
        return next(iter(uploaded.keys()))

    # 2. Local desktop environment -> try native file-picker popup
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askopenfilename(
            title="Select Medical Image (X-Ray / MRI)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All Files", "*.*")]
        )
        root.destroy()
        if picked:
            return picked
    except Exception:
        pass

    # 3. Fallback -> manual path entry in terminal
    manual_path = input("Enter the full path to your X-ray or MRI image: ").strip().strip('"').strip("'")
    if not os.path.isfile(manual_path):
        raise FileNotFoundError(f"Could not find image at: {manual_path}")
    return manual_path


def load_and_preprocess(image_path, size=(256, 256)):
    """Loads any grayscale/RGB/16-bit medical scan, converts to grayscale, resizes, and normalizes to [0, 1]."""
    pil_img = Image.open(image_path)
    print(f"Loaded: {os.path.basename(image_path)} | Original Size: {pil_img.size} | Mode: {pil_img.mode}")

    # Handle 16-bit grayscale scans safely before resizing
    if pil_img.mode in ("I", "I;16"):
        arr = np.asarray(pil_img, dtype=np.float64)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    elif pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    pil_resized = pil_img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(pil_resized, dtype=np.float64) / 255.0


# ==============================================================================
# STAGE 3 & 4: NOISE REMOVAL (Median + Gaussian) & CLAHE ENHANCEMENT
# ==============================================================================
def denoise_and_enhance(img):
    denoised = gaussian_filter(median_filter(img, size=3), sigma=0.8)
    u8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    if HAS_CV2:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8) / 255.0
    else:
        hist, _ = np.histogram(u8.flatten(), bins=256, range=(0, 256))
        clip = int(2.0 * (u8.size / 256.0))
        excess = np.maximum(hist - clip, 0).sum()
        hist = np.minimum(hist, clip) + (excess // 256)
        cdf = np.cumsum(hist).astype(np.float64)
        enhanced = ((cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8))[u8]
    return denoised, enhanced


# ==============================================================================
# STAGE 5: NOVEL SPATIAL + HISTOGRAM-PEAK K-MEANS (From Scratch)
# ==============================================================================
def spatial_peak_kmeans(img, K, alpha=0.15, max_iters=100, tol=1e-4):
    H, W = img.shape
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    X = np.column_stack([img.ravel(), (alpha * xx).ravel(), (alpha * yy).ravel()])

    # Deterministic Histogram-Peak Seeding
    hist, edges = np.histogram(img.ravel(), bins=256, range=(0.0, 1.0))
    smooth = gaussian_filter1d(hist.astype(np.float64), sigma=4.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks = sorted(
        [i for i in range(1, 255) if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]],
        key=lambda i: smooth[i],
        reverse=True,
    )
    chosen = []
    for p in peaks:
        if all(abs(centers[p] - c) >= (0.65 / K) for c in chosen):
            chosen.append(centers[p])
        if len(chosen) == K:
            break
    if len(chosen) < K:
        chosen = list(np.quantile(img, np.linspace(0.08, 0.92, K)))

    centroids = np.array([X[np.argmin(np.abs(X[:, 0] - val))] for val in sorted(chosen)])

    for it in range(1, max_iters + 1):
        dist_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        wcss = float(np.sum(np.min(dist_sq, axis=1)))

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(K)
        ])
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Sort clusters from darkest (air/background) to brightest (bone/dense tissue)
    order = np.argsort(centroids[:, 0])
    centroids = centroids[order]
    inv = np.zeros_like(order)
    inv[order] = np.arange(K)
    labels = inv[labels].reshape(H, W)
    return labels, centroids, wcss, it


# ==============================================================================
# STAGE 6: MORPHOLOGICAL POST-PROCESSING
# ==============================================================================
def morphological_cleanup(label_map, K):
    selem = np.ones((3, 3), dtype=bool)
    cleaned = np.zeros_like(label_map)
    for j in range(K):
        mask = binary_closing(binary_opening(label_map == j, structure=selem), structure=selem)
        cleaned[mask] = j
    return cleaned


# ==============================================================================
# STAGE 7: EXECUTION, METRICS TABLE & 2 REPORT FIGURES
# ==============================================================================
if __name__ == "__main__":
    selected_path = get_image_path(IMAGE_PATH)
    raw_img = load_and_preprocess(selected_path, size=IMAGE_SIZE)
    _, enh_img = denoise_and_enhance(raw_img)

    runs = {}
    print(f"\n{'K':<3} | {'Iters':<5} | {'WCSS (J)':<10} | {'PSNR (dB)':<9} | {'Cluster Area Breakdown (%)'}")
    print("-" * 72)

    for K in K_LIST:
        raw_labels, cents, wcss, iters = spatial_peak_kmeans(enh_img, K=K, alpha=ALPHA)
        clean_labels = morphological_cleanup(raw_labels, K=K)
        recon = cents[clean_labels, 0]
        psnr = 10.0 * np.log10(1.0 / (np.mean((enh_img - recon) ** 2) + 1e-12))
        areas = [round(100.0 * np.mean(clean_labels == j), 1) for j in range(K)]

        runs[K] = {"raw": raw_labels, "clean": clean_labels, "wcss": wcss, "psnr": psnr, "iters": iters}
        print(f"{K:<3} | {iters:<5} | {wcss:<10.2f} | {psnr:<9.2f} | {areas}")

    # FIGURE 1: 4-Panel Workflow Progression (at DEFAULT_K)
    fig1, ax1 = plt.subplots(1, 4, figsize=(14, 3.6))
    panels1 = [
        (raw_img, "1. Input Medical Scan", "gray"),
        (enh_img, "2. Denoised + CLAHE", "gray"),
        (runs[DEFAULT_K]["raw"], f"3. K-Means Output (K={DEFAULT_K})", "nipy_spectral"),
        (runs[DEFAULT_K]["clean"], "4. Post-Processed Final", "nipy_spectral"),
    ]
    for ax, (data, title, cmap) in zip(ax1, panels1):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("fig1_pipeline.png", dpi=300)

    # FIGURE 2: Multi-K Comparison (K=2, 3, 4) + WCSS Elbow Curve
    fig2, ax2 = plt.subplots(1, 4, figsize=(14, 3.6))
    for idx, K in enumerate([2, 3, 4]):
        ax2[idx].imshow(runs[K]["clean"], cmap="nipy_spectral")
        ax2[idx].set_title(f"Segmented (K={K})", fontsize=10, fontweight="bold")
        ax2[idx].axis("off")

    ax2[3].plot(K_LIST, [runs[k]["wcss"] for k in K_LIST], "o-", color="#1f77b4", linewidth=2)
    ax2[3].set_title("Elbow Curve (WCSS vs K)", fontsize=10, fontweight="bold")
    ax2[3].set_xlabel("Clusters (K)")
    ax2[3].set_ylabel("WCSS (J)")
    ax2[3].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig2_k_analysis.png", dpi=300)
    plt.show()