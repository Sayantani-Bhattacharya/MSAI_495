import numpy as np
import cv2 as cv

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

    # FIRST PASS: assign temporary labels in sequence and record equivalences
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

    # SECOND PASS: update labels using resolved equivalence dictionary.
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if label_img[i, j] > 0:
                label_img[i, j] = label_map[label_img[i, j]]

    # Remove padding
    label_img = label_img[1:-1, 1:-1]

    # # Count unique labels
    unique_labels = np.unique(label_img[label_img > 0])
    # num_components = len(unique_labels)

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
    
    # Filter out small components based on pixel count threshold
    
    # Count the number of pixels in each component.
    count = {}
    for i in range(rows):
        for j in range(cols):
            label = label_img[i, j]
            if label > 0:
                count[label] = count.get(label, 0) + 1

    # Filter components based on size
    size_threshold = 500
    filtered_labels = {label for label, size in count.items() if size >= size_threshold}
    return filtered_labels


if __name__ == "__main__":
    print("Connect Component Labeling")
    connect_component_labeling('gun.bmp')
    print("Image processing completed.")  # type: ignore
