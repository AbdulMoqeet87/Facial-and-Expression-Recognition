import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load face images
image_paths = [
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.1.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.2.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.3.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.4.png',
    r'E:\3rd semester\Linear Algebra\Python_Assignment\yalefaces\subject01\subject1.5.png',
    
    # Add more paths as needed
]

# Load images and flatten them
face_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten().resize(60,60) for path in image_paths]

# Combine the flattened images into a data matrix
face_data_matrix = np.vstack(face_images)

# Transpose the data matrix if you want each column to represent a feature
face_data_matrix = face_data_matrix.T

# Center the data by subtracting the mean
mean_face = np.mean(face_data_matrix, axis=0)
centered_face_data = face_data_matrix - mean_face

# Perform PCA on face data
pca_faces = PCA()
pca_faces.fit(centered_face_data)

# Display the first few eigenfaces
num_eigenfaces_to_display = 5
eigenfaces = pca_faces.components_[:num_eigenfaces_to_display]

# Reshape and display eigenfaces
plt.figure(figsize=(12, 6))
for i in range(num_eigenfaces_to_display):
    eigenface = eigenfaces[i]
    eigenface_image = eigenface.reshape(face_images[0].shape[0], face_images[0].shape[1])

    plt.subplot(1, num_eigenfaces_to_display, i + 1)
    plt.imshow(eigenface_image, cmap='gray')
    plt.title(f"Eigenface {i + 1}")

plt.show()