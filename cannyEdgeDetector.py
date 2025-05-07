import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


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

def load_grayscale(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

if __name__ == "__main__":
    # image_paths = ["hw5/test1.bmp"]
    image_paths = ["hw5/joy1.bmp", "hw5/lena.bmp", "hw5/pointer1.bmp", "hw5/test1.bmp"]


    for path in image_paths:
        I = load_grayscale(path)
        S = gausian_smoothening(I, N=5, sigma=0.6)

        # Save the gausian smoothed image
        smoothed_image = Image.fromarray(S).convert("L")
        smoothed_image.save(f"smoothed_{path.split('/')[-1]}")

        Mag, Theta = calculateImgGrad(S)
        print(f"[{path}] Mag stats: min={Mag.min():.2f}, max={Mag.max():.2f}, mean={Mag.mean():.2f}")
        T_low, T_high = find_threshold(Mag, percentageOfNonEdge=0.7)
        print(f"T_low = {T_low:.2f}, T_high = {T_high:.2f}")

        T_high = max(T_high, 50) # Ensure T_high is not too low
        T_low = min(T_high, 30) # Ensure T_low is not too low

        print(f"T_low = {T_low:.2f}, T_high = {T_high:.2f}")


        Mag_suppressed = nonmaxima_suppress(Mag, Theta)
        edges = edge_linking(Mag_suppressed, T_low, T_high)

        # Save the result
        result_image = Image.fromarray(edges)
        result_image.save(f"output_{path.split('/')[-1]}")
        print(f"Processed {path} and saved the result.")

        # Compare with OpenCV
        I_uint8 = I.astype(np.uint8)

        # Sobel
        sobelx = cv.Sobel(I_uint8, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(I_uint8, cv.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        Image.fromarray(sobel).save(f"sobel_{path.split('/')[-1]}")

        # OpenCV Canny
        canny = cv.Canny(I_uint8, 100, 200)
        Image.fromarray(canny).save(f"canny_cv_{path.split('/')[-1]}")




# For my clarity:

# We perform non-maxima suppression to thin out edges and retain only the most significant edge pixels along the gradient direction. This step is crucial in edge detection algorithms like Canny because it ensures that the detected edges are sharp and precise, rather than thick or blurry.

# Why is Non-Maxima Suppression Necessary?
# Thinning Edges:

# Without non-maxima suppression, edges in the gradient magnitude image may appear thick because multiple neighboring pixels may have high gradient values.
# Non-maxima suppression ensures that only the pixel with the highest gradient magnitude in the direction of the gradient is retained, resulting in a single-pixel-wide edge.
# Improving Edge Localization:

# By suppressing non-maximum pixels, the algorithm improves the accuracy of edge localization, making it easier to identify the exact position of edges.
# Reducing Noise:

# Non-maxima suppression helps reduce noise by discarding weaker gradient values that are not part of the true edge.

# What Happens if the Image Has Thick Boundaries?
# If the image has thick boundaries (e.g., due to blurring or low resolution), non-maxima suppression will still thin the edges to a single-pixel width. However:

# Thick Boundaries Before Suppression:

# Thick boundaries in the gradient magnitude image occur when the intensity changes gradually over a region rather than sharply at a single point.

# Impact on Thick Boundaries:

# If the boundary is inherently thick (e.g., due to the nature of the object or image resolution), non-maxima suppression may result in a single edge line that approximates the center of the thick boundary.
# This is desirable in most edge detection tasks, as it simplifies the representation of edges.