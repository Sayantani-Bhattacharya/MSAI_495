import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def equalizeHist(image):
    # Calculate histogram
    hist = cv.calcHist([image], [0], None, [256], [0, 256])    
    # Normalize histogram
    hist = hist / hist.sum()    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()    
    # Normalize CDF to range [0, 255]
    cdf_normalized = (cdf * 255).astype(np.uint8)    
    # Map pixel values using the CDF
    equalized_image = cdf_normalized[image]    
    return equalized_image

def plot_histogram(image, title):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Pixel Count")
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

def lighting_correction(image, method='linear'):
    """
    Correct gross shading by fitting either:
      - a plane:     I_shade(u,v) = a1*u + a2*v + a3          (method='linear')
      - a quadratic: I_shade(u,v) = a1*u^2 + a2*v^2 + a3*u*v + a4*u + a5*v + a6
    and then subtracting the fitted shading surface from the image.
    """

    rows, cols = image.shape
    # v = row index, u = col index
    v, u = np.mgrid[0:rows, 0:cols]
    u_flat = u.ravel().astype(np.float64)
    v_flat = v.ravel().astype(np.float64)
    t = image.ravel().astype(np.float64)  # intensity vector

    if method == 'linear':
        # build A = [u v 1]
        A = np.vstack([u_flat, v_flat, np.ones_like(u_flat)]).T
    elif method == 'quadratic':
        # build A = [u^2, v^2, u*v, u, v, 1]
        A = np.vstack([
            u_flat**2,
            v_flat**2,
            u_flat * v_flat,
            u_flat,
            v_flat,
            np.ones_like(u_flat)
        ]).T
    else:
        raise ValueError("Unknown lighting correction method.")

    # solve for a via normal equations (least squares)
    #    a = (A^T A)^{-1} A^T t
    ATA = A.T @ A
    ATA_inv = np.linalg.inv(ATA)
    a = ATA_inv @ (A.T @ t)

    # compute fitted shading surface
    S_flat = A @ a
    S = S_flat.reshape(rows, cols)

    # subtract shading and re-normalize contrast
    corrected = image.astype(np.float64) - S
    # add back the mean so we keep overall brightness
    corrected += np.mean(S)
    # clip & convert back to uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

if __name__ == "__main__":
    image = cv.imread('hw3/moon.bmp', cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Plot histogram before equalization
    plot_histogram(image, "Histogram Before Equalization")

    # Apply histogram equalization
    final_img = equalizeHist(image)
    cv.imwrite('output.png', final_img)

    # Plot histogram before equalization
    plot_histogram(final_img, "Histogram After Equalization")

    # Apply lighting correction
    corrected_image_linear = lighting_correction(final_img, method='linear')
    cv.imwrite('linear_corrected_output.png', corrected_image_linear)
  
    corrected_image_quadratic = lighting_correction(final_img, method='quadratic')
    cv.imwrite('quadratic_corrected_output.png', corrected_image_quadratic)