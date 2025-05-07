import numpy as np
from PIL import Image, ImageDraw
import os
import math
import cv2 as cv
import matplotlib.pyplot as plt

def load_grayscale(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def gaussian_kernel(N, sigma):
    ax = np.arange(-N // 2 + 1., N // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # To Ensure the sum of all kernel values equals 1. This prevents the image's overall brightness from changing during convolution.
    return kernel / np.sum(kernel)

def conv2d(image, kernel):
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)
    return output

def gausian_smoothening(I, N=5, sigma=1.4):
    G = gaussian_kernel(N, sigma)
    return conv2d(I, G)

def save_png(image, path):
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def hough_transform(edge_img, rho_res=1, theta_res=0.4):
    h, w = edge_img.shape
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    thetas = np.deg2rad(np.arange(0, 180, theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    y_idxs, x_idxs = np.nonzero(edge_img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            rho_idx = rho + diag_len
            if 0 <= rho_idx < len(rhos):
                accumulator[rho_idx, t_idx] += 1
    return accumulator, rhos, thetas

def get_peak_lines(accumulator, rhos, thetas, threshold):
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] > threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append((rho, theta))
    return lines

def draw_lines(image, lines):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=1)
    return image

def calculateImgGrad(I):
    # Sobel operators
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    Ix = conv2d(I, Gx)
    Iy = conv2d(I, Gy)

    Mag = np.hypot(Ix, Iy)
    Theta = np.arctan2(Iy, Ix)
    return Mag, Theta

def nonmaxima_suppress(Mag, Theta):
    M, N = Mag.shape # Same as image shape.
    # Every pixel has an gradient angle, and if at that angle any one of the 2 neighboring pixels have a higher gradient magnitude, 
    # then that pixel is not a maxima and thus also not an edge.
    Z = np.zeros((M, N), dtype=np.float32)
    angle = np.rad2deg(Theta) % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q, r = 255, 255
                a = angle[i, j]

                # Determine neighbors in gradient direction
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = Mag[i, j+1]
                    r = Mag[i, j-1]
                elif (22.5 <= a < 67.5):
                    q = Mag[i+1, j-1]
                    r = Mag[i-1, j+1]
                elif (67.5 <= a < 112.5):
                    q = Mag[i+1, j]
                    r = Mag[i-1, j]
                elif (112.5 <= a < 157.5):
                    q = Mag[i-1, j-1]
                    r = Mag[i+1, j+1]

                if Mag[i, j] >= q and Mag[i, j] >= r:
                    Z[i, j] = Mag[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z

def find_threshold(Mag, percentageOfNonEdge=0.7):
    flat = Mag.flatten()
    sorted_mag = np.sort(flat)
    # So basically if the percentage of non-edge pixels is 0.7, then we take the 70th percent-th value of 
    # the gradient magnitude pixel values as the high threshold.
    # And value of gradient magnitude corresponds in a way to edge of the image.
    # Mag value say how much of this pixel is likely to be an edge.
    thresh_idx = int(len(sorted_mag) * percentageOfNonEdge)
    T_high = sorted_mag[thresh_idx]
    T_low = 0.5 * T_high
    return T_low, T_high

def edge_linking(Mag, T_low, T_high):
    strong, weak = 255, 75
    res = np.zeros_like(Mag, dtype=np.uint8)
    strong_i, strong_j = np.where(Mag >= T_high)
    weak_i, weak_j = np.where((Mag <= T_high) & (Mag >= T_low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Track edges
    M, N = res.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if res[i, j] == weak:
                if np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res

def canny_edge_detector(S):
    # Save the gausian smoothed image
    smoothed_image = Image.fromarray(S).convert("L")
    # smoothed_image.save(f"smoothed_{path.split('/')[-1]}")

    Mag, Theta = calculateImgGrad(S)
    # print(f"[{path}] Mag stats: min={Mag.min():.2f}, max={Mag.max():.2f}, mean={Mag.mean():.2f}")
    T_low, T_high = find_threshold(Mag, percentageOfNonEdge=0.7)
    # print(f"T_low = {T_low:.2f}, T_high = {T_high:.2f}")

    T_high = max(T_high, 50) # Ensure T_high is not too low
    T_low = min(T_high, 30) # Ensure T_low is not too low

    # print(f"T_low = {T_low:.2f}, T_high = {T_high:.2f}")


    Mag_suppressed = nonmaxima_suppress(Mag, Theta)
    edges = edge_linking(Mag_suppressed, T_low, T_high)
    return edges

def plot_hough_accumulator(accumulator, rhos, thetas):
    plt.figure(figsize=(10, 8))
    plt.imshow(
        accumulator,
        extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]],
        cmap='hot',
        aspect='auto'
    )
    plt.title('Hough Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar(label='Votes')
    plt.grid(False)
    return plt


if __name__ == "__main__":
    image_paths = ["hw6/input.bmp", "hw6/test.bmp", "hw6/test2.bmp"]

    for path in image_paths:
        img_gray = load_grayscale(path)
        S = gausian_smoothening(img_gray, N=5, sigma=0.6)

        # Custom canny edge detector
        edges = canny_edge_detector(S)      
        acc, rhos, thetas = hough_transform(edges)

        # Threshold for significant lines - tweak based on image
        peak_lines = get_peak_lines(acc, rhos, thetas, threshold=50)
        # Plot Hough accumulator
        plt = plot_hough_accumulator(acc, rhos, thetas)
        

        # Draw lines on original (RGB) image
        original_img = Image.open(path).convert("RGB")
        result_img = draw_lines(original_img, peak_lines)

        base = os.path.splitext(os.path.basename(path))[0]
        # Save the accumulator image
        save_png(edges, f"{base}_edges.png")
        result_img.save(f"{base}_lines.png")
        plt.savefig(f"{base}_hough_accumulator.png")

        print(f"{path}: saved edges and line-detected images. Lines found: {len(peak_lines)}")
