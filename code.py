import numpy as np
import cv2 as cv

# Morphological Operators.
def dilation(image, kernel_size=(3, 3), kernel = None, iterations=1):

    # Add an option to define the kernel matrix itself.
    # kernel = [(1,1,1,1), (1,0,0,1), (1,0,0,1), (1,1,1,1)]
    kernel = [(0,0,0,0), (0,1,1,0), (0,1,1,0), (0,0,0,0)]


    if kernel is not None:
        print("Using custom kernel")
        kernel = np.array(kernel, dtype=np.uint8)
    else:
        print("Using default kernel")
        kernel = np.ones(kernel_size, dtype=np.uint8)

    # Threshold image to binary (ensure values are 0 or 255)
    _, binary_img = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Structuring element (assume all 1s in a square)
    k_rows, k_cols = kernel.shape
    k_center_r, k_center_c = k_rows // 2, k_cols // 2

    # Pad the input image
    padded_img = np.pad(binary_img, ((k_center_r, k_center_r), (k_center_c, k_center_c)), mode='constant', constant_values=0)

    # Prepare output image
    dilated_img = np.zeros_like(binary_img)

    # Dilation operation
    # Assuming `kernel` is a 2D array of the same size as the neighborhood
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            # Extract the neighborhood region
            neighborhood = padded_img[i:i + k_rows, j:j + k_cols]
            # Apply the kernel to the neighborhood
            if np.any(neighborhood * kernel == 255):  # Element-wise multiplication
                dilated_img[i, j] = 255

    return dilated_img

def erosion(image, kernel_size=(3, 3), iterations=1):
    # Threshold image to binary (ensure values are 0 or 255)
    _, binary_img = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Define structuring element (3x3 square by default)
    kernel = np.ones(kernel_size, dtype=np.uint8)
    rows, cols = binary_img.shape

    # Structuring element (assume all 1s in a square)
    k_rows, k_cols = kernel_size
    k_center_r, k_center_c = k_rows // 2, k_cols // 2

    # Pad the input image
    padded_img = np.pad(binary_img, ((k_center_r, k_center_r), (k_center_c, k_center_c)), mode='constant', constant_values=0)

    # Prepare output image
    erroded_img = np.zeros_like(binary_img)

    # Dilation operation
    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood region
            neighborhood = padded_img[i:i + k_rows, j:j + k_cols]
            # If all value in the neighborhood is 255, set output to 255

            # if all 255 only then 255.
            if np.all(neighborhood == 255):
                erroded_img[i, j] = 255
            # np.all

    return erroded_img

def opening(image, kernel_size=(3, 3)):
    # Apply erosion followed by dilation
    eroded_img = erosion(image, kernel_size)
    opened_img = dilation(eroded_img, kernel_size)
    return opened_img

def closing(image, kernel_size=(3, 3)):
    # Apply dilation followed by erosion
    dilated_img = dilation(image, kernel_size)
    closed_img = erosion(dilated_img, kernel_size)
    return closed_img

def boundary(image, kernel_size=(3, 3), iterations=1):
    # Apply dilation and then erosion
    dilated_img = dilation(image, kernel_size, iterations)
    eroded_img = erosion(image, kernel_size, iterations)

    # Boundary is the difference between dilated and eroded images
    boundary_img = dilated_img - eroded_img
    return boundary_img


def connect_component_labeling(binary_image):
    print("Reading image...")
    
    # Load the images from bmp file
    image = cv.imread(binary_image, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to binary (0 = background, 1 = foreground)
    _, image = cv.threshold(image, 127, 1, cv.THRESH_BINARY)
    rows, cols = image.shape

    # Padded image to handle edge pixels.
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    label_img = np.zeros_like(padded_image, dtype=int)
    current_label = 1
    # This dictionary stores each label and the set of labels it's equivalent to.
    equivalence_dict = {}

    # FIRST PASS: Assign temporary labels in sequence and record equivalences
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Background elements.
            if padded_image[i, j] == 0:
                continue

            neighbors = [label_img[i, j - 1], label_img[i - 1, j]]  # Left and Top
            non_zero_neighbors = [label for label in neighbors if label > 0]

            # If no neighbors, assign new label
            if not non_zero_neighbors:
                label_img[i, j] = current_label
                equivalence_dict[current_label] = {current_label}
                current_label += 1
            else:
                # If neighbors exist, assign the smallest label which is also the top label.
                min_label = min(non_zero_neighbors)
                label_img[i, j] = min_label
                for neighbor_label in non_zero_neighbors:
                    # If neighbor_label is not yet in equivalence_dict, it gets just {neighbor_label} (a set with itself).
                    equivalence_dict[min_label].update(equivalence_dict.get(neighbor_label, {neighbor_label}))
                    equivalence_dict[neighbor_label] = equivalence_dict[min_label]

    # Flatten equivalence classes: map every label to the smallest label in its equivalence set.
    label_map = {}
    for eq_class in equivalence_dict.values():
        root = min(eq_class)
        for label in eq_class:
            label_map[label] = root

    # SECOND PASS: Update labels using resolved equivalence dictionary.
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if label_img[i, j] > 0:
                label_img[i, j] = label_map[label_img[i, j]]

    # Remove padding
    label_img = label_img[1:-1, 1:-1]

    # Print label matrix to file
    # with open('label_matrix.txt', 'w') as f:
    #     for row in label_img:
    #         f.write(' '.join(map(str, row)) + '\n')

    # Filer
    size_filtered_labels = size_filter(rows, cols, label_img)
    # Count unique labels
    unique_labels = np.unique(list(size_filtered_labels))
    num_components = len(size_filtered_labels)

    # Generate random color for each label
    label_color_map = {label: np.random.randint(0, 255, size=3) for label in unique_labels}
    color_output = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if label_img[i, j] > 0:
                # Check if the label value of label_img[i, j] is in label_color_map
                if label_img[i, j] in label_color_map:
                    color_output[i, j] = label_color_map[label_img[i, j]]

    # Save colored output
    cv.imwrite('labeled_output.bmp', color_output)
    print(f"Total connected components: {num_components}")
    return label_img, num_components

def size_filter(rows, cols, label_img):   
    # Filter out small components based on pixel count threshold.   
    # Count the number of pixels in each component.
    count = {}
    for i in range(rows):
        for j in range(cols):
            label = label_img[i, j]
            if label > 0:
                count[label] = count.get(label, 0) + 1
    size_threshold = 500
    filtered_labels = {label for label, size in count.items() if size >= size_threshold}
    return filtered_labels

if __name__ == "__main__":
    # print("Connect Component Labeling")
    # connect_component_labeling('gun.bmp')
    # print("Image processing completed.")  # type: ignore

    # Load image in grayscale
    # TODO: Add function to load image

    image = cv.imread('testImg2/palm.bmp', cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Apply Dilation
    dilated_img = dilation(image, kernel_size=(4, 4), iterations=1) 
    # TODO: Add function to print-out the image.
    # Save and return
    cv.imwrite('dilated_output.png', dilated_img)   

    # Apply Erosion
    eroded_img = erosion(image, kernel_size=(3, 3), iterations=1)
    cv.imwrite('eroded_output.png', eroded_img)

    # Apply Boundary
    boundary_img = boundary(image, kernel_size=(2, 2), iterations=1)
    cv.imwrite('boundary_output.png', boundary_img)

    # Apply Opening
    opened_img = opening(image, kernel_size=(3, 3))
    cv.imwrite('opened_output.png', opened_img)

    # Apply Closing
    closed_img = closing(image, kernel_size=(3, 3))
    cv.imwrite('closed_output.png', closed_img)



    



