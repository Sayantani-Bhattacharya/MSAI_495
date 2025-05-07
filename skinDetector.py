import numpy as np
import cv2
import os

def save_histogram(hist, filename):
    """
    Save the histogram to a file.
    """ 
    hist_img = (hist * 255).astype(np.uint8)
    hist_img = cv2.normalize(hist_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(filename, hist_img)

def plot_histogram(hist):
    """
    Plot the histogram using matplotlib.
    """
    import matplotlib.pyplot as plt
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.xlabel('S Channel')
    plt.ylabel('H Channel')
    plt.colorbar()
    plt.savefig("trained_histogram.png")  

def train_histogram(train_img_paths, bins=256):
    """
    1) Load a BMP image.
    2) Let the user draw one or more ROIs [Region of Interest] that contain only skin.
    3) Build and normalize a 2D histogram over the H-S plane.
    """
    hist = np.zeros((bins, bins), dtype=np.float32)

    for train_img_path in train_img_paths:
        img_bgr = cv2.imread(train_img_path)
        if img_bgr is None:
            print(f"Warning: Cannot read {train_img_path}")
            continue

        # To use RGB instead of HSV:
        # img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        while True:
            rect = cv2.selectROI(f"Select skin region in {train_img_path} (Press ENTER to accept, ESC to skip)", img_bgr, showCrosshair=True)
            if rect == (0, 0, 0, 0):
                break  # If user cancels selection (ESC)

            x, y, w, h = map(int, rect)
            roi = img_hsv[y:y+h, x:x+w]

            H = roi[:, :, 0].ravel()
            S = roi[:, :, 1].ravel()

            # Update histogram
            roi_hist, _, _ = np.histogram2d(
                H, S,
                bins=bins,
                range=[[0, 255], [0, 255]]
            )
            hist += roi_hist.astype(np.float32)

    # Normalize after collecting all ROIs
    hist /= hist.sum()
    save_histogram(hist, "trained_histogram.png")

    return hist

def kmeans_threshold(hist, k=2):
    """
    1) Flatten histogram into a list of probabilities: where rosa are places next to each other.
    2) Apply simple 1D K-means clustering to separate into K groups.
    3) Pick the cluster with highest mean as 'skin'.
    """
    values = hist.ravel()
    nonzero_idx = np.nonzero(values)[0]
    data = values[nonzero_idx].reshape(-1, 1)

    # Randomly initialize K centers
    centers = np.random.choice(data.flatten(), k).reshape(k, 1)

    for _ in range(10): 
        # Assign points to closest center
        dists = np.abs(data - centers.T)
        labels = np.argmin(dists, axis=1)

        # Update centers
        for i in range(k):
            if np.any(labels == i):
                centers[i] = data[labels == i].mean()

    # Find the cluster with the highest center (assume skin has higher prob)
    skin_cluster = np.argmax(centers)

    # Create threshold mask
    skin_idx = nonzero_idx[labels == skin_cluster]
    threshold_mask = np.zeros_like(values, dtype=np.uint8)
    threshold_mask[skin_idx] = 1

    return threshold_mask.reshape(hist.shape)

def detect_skin(img_path, hist_mask):
    """
    1) Read image.
    2) For each pixel, use (H,S) to check if it falls in the skin cluster.
    3) Output the skin mask and masked image.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Skipping {img_path}: file not found.")
        return
    
    # To use RGB instead of HSV:
    # img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]

    # Lookup from mask
    mask = hist_mask[H, S]
    mask = (mask > 0).astype(np.uint8) * 255

    skin_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(f"{base}_mask.png", mask)
    cv2.imwrite(f"{base}_skin.png", skin_only)

    print(f"Saved {base}_mask.png and {base}_skin.png")

if __name__ == "__main__":
    file = "histogram.csv"
    # Train histogram from skin region
    hist = train_histogram(["skinData/1.png", "skinData/2.png", "skinData/3.png","skinData/4.png", "skinData/5.png", "skinData/6.png", "skinData/7.png"])

    # Save the histogram in a file so not needed to train again.
    np.savetxt(file, hist, delimiter=",")

    # Load the histogram from the file
    hist = np.loadtxt(file, delimiter=",")

    plot_histogram(hist)
    
    # K-means thresholding
    hist_mask = kmeans_threshold(hist, k=2)

    # Detect skin in test images
    for fn in ["hw4/joy1.bmp", "hw4/gun1.bmp", "hw4/pointer1.bmp"]:
        detect_skin(fn, hist_mask)

