import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from fir_conv import fir_conv
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough

start_total = time.time()

# ------------------------------------------------
# 1. Load image and convert to grayscale [0,1]
# ------------------------------------------------
start = time.time()
rgb_img = cv2.cvtColor(cv2.imread('basketball_large.png'), cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) / 255.0
print(f"[1] Image loaded and converted: {time.time() - start:.2f}s")

# ------------------------------------------------
# 2. Sobel edge detection for various threshold values
# ------------------------------------------------
start = time.time()
thres_vals = [0.05, 0.1, 0.15, 0.2, 0.25]
sobel_outputs = []
edge_counts = []

for thres in thres_vals:
    edge_img = sobel_edge(gray_img, thres)
    sobel_outputs.append(edge_img)
    edge_counts.append(np.sum(edge_img))
print(f"[2] Sobel edge detection (all thresholds): {time.time() - start:.2f}s")

# Display results of Sobel for each threshold
fig, axs = plt.subplots(1, len(thres_vals), figsize=(15, 3))
for i, thres in enumerate(thres_vals):
    axs[i].imshow(sobel_outputs[i], cmap='gray')
    axs[i].set_title(f"Sobel\nthres={thres}")
    axs[i].axis('off')
plt.tight_layout()
plt.savefig('sobel_outputs.png')
plt.show()

# Plot number of detected edges vs threshold
plt.plot(thres_vals, edge_counts, marker='o')
plt.title("Edges vs Threshold (Sobel)")
plt.xlabel("Threshold")
plt.ylabel("Detected edge pixels")
plt.grid(True)
plt.savefig('sobel_threshold_plot.png')
plt.show()

# ------------------------------------------------
# 3. LoG edge detection
# ------------------------------------------------
start = time.time()
log_result = log_edge(gray_img)
print(f"[3] LoG edge detection: {time.time() - start:.2f}s")

plt.imshow(log_result, cmap='gray')
plt.title("LoG Edge Detection")
plt.axis('off')
plt.savefig('log_output.png')
plt.show()

# --------------------------------------------------
# 4. circ_hough using Sobel edge result
# --------------------------------------------------
start = time.time()
binary_edge = sobel_edge(gray_img, thres=0.1)

# Define Hough accumulator space dimensions
dim = np.array([50, 50, 20])  # [x, y, radius] bins
R_max = 60.0
vmin_values = [2, 4, 6, 8, 10]  # v_min thresholds to filter weak circles

rows, cols = gray_img.shape
nx, ny, nr = dim

print(f"[4] Hough via Sobel:")
for v_min in vmin_values:
    t0 = time.time()

    # Run Hough circle detection
    centers, radii, hough_space = circ_hough(binary_edge, R_max, dim, v_min)

    print(f"  ► v_min={v_min} | Circles: {len(centers)} | Time: {time.time() - t0:.2f}s")

    # --------------------------------
    # Compute votes for each detected circle
    # --------------------------------
    votes = []
    for (x, y), r in zip(centers, radii):
        a_idx = int(np.round((x / (cols - 1)) * (nx - 1)))
        b_idx = int(np.round((y / (rows - 1)) * (ny - 1)))
        r_idx = int(np.round(((r - 1) / (R_max - 1)) * (nr - 1)))

        if 0 <= a_idx < nx and 0 <= b_idx < ny and 0 <= r_idx < nr:
            votes.append(hough_space[b_idx, a_idx, r_idx])
        else:
            votes.append(0)

    # --------------------------------
    # Keep only top-N strongest circles
    # --------------------------------
    N = 15
    strongest = np.argsort(votes)[-N:]  # indices of N circles with highest votes

    img_copy = rgb_img.copy()
    for i in strongest:
        x, y = centers[i]
        r = radii[i]
        cv2.circle(img_copy, (int(x), int(y)), int(r), (255, 0, 0), 2)

    # --------------------------------
    # Save and show result
    # --------------------------------
    plt.imshow(img_copy)
    plt.title(f"Hough (Sobel) v_min={v_min}")
    plt.axis('off')
    plt.savefig(f"hough_sobel_vmin_{v_min}.png")
    plt.show()

print(f"[4] Total time for Sobel-based Hough: {time.time() - start:.2f}s")
