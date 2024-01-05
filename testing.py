import os
import cv2
import numpy as np
from sklearn.decomposition import PCA

# Function to load images from a directory
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.flatten())  # Flatten the image to a 1D array
    return images

# Function to perform PCA and retain 99% energy
def perform_pca(images):
    pca = PCA(0.99)  # Retain 99% energy
    pca.fit(images)
    return pca.components_

# Function to recognize a person in a test image
def recognize_person(test_image, pca_bases):
    test_image_flattened = test_image.flatten()

    min_loss = float('inf')
    recognized_person = None

    for i, pca_basis in enumerate(pca_bases):
        # Project test image onto PCA basis
        projection = np.dot(test_image_flattened, pca_basis)

        # Back project the projection to original space
        back_projected_image = np.dot(projection, pca_basis.T)

        # Calculate loss between the original image and back-projected image
        loss = np.linalg.norm(test_image_flattened - back_projected_image)

        # Update recognized person if loss is smaller
        if loss < min_loss:
            min_loss = loss
            recognized_person = i + 1  # Person numbering starts from 1

    return recognized_person

# Main function
def main():
    yalefaces_dir = "path/to/yalefaces"
    n_train_images = 10
    n_test_images = 1

    # Load images for training
    training_data = []
    for subject in range(1, 16):
        subject_dir = os.path.join(yalefaces_dir, f"subject{subject}")
        subject_images = load_images(subject_dir)
        np.random.shuffle(subject_images)  # Shuffle images for randomness
        training_data.extend(subject_images[:n_train_images])

    # Perform PCA on training data
    pca_bases = perform_pca(training_data)

    # Test with sample images
    test_images_dir = "path/to/test/images"
    for filename in os.listdir(test_images_dir):
        test_img_path = os.path.join(test_images_dir, filename)
        test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

        # Recognize the person in the test image
        recognized_person = recognize_person(test_image, pca_bases)

        # Visualize the test image and recognized person
        cv2.imshow(f"Test Image - Recognized Person {recognized_person}", test_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
t