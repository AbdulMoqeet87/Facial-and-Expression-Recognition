import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_image(file_path):
    img=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(50,50))
    return img.flatten()




all_images_list=[]
label_list=[]

express_folder=r'C:\Users\DELL\Downloads\data\yalefaces'
for expression in["happy","sad","normal","sleepy","surprised","wink"]:
    dataset=[]
    label=[]
    for person_id in range(1, 11):
        person_folder=os.path.join(express_folder,rf"subject{person_id:02d}\subject{person_id:02d}")
        img_filename=f"{person_folder}.{expression}.png"
        
        img_path = os.path.join(express_folder, person_folder,img_filename)
        img= read_image(img_path)
        dataset.append(img)
        label.append(expression)
    all_images_list.append(np.array(dataset))
    label_list.append(np.array(label))
    
        



images_resized = all_images_list


data_matrices = [np.vstack([img.flatten() for img in subject_images]) for subject_images in images_resized]
all_mean_vectors = [np.mean(data_matrix, axis=0) for data_matrix in data_matrices]
centered_data_matrices = [data_matrix - mean_vector for data_matrix, mean_vector in zip(data_matrices, all_mean_vectors)]



covariance_matrices = [np.cov(centered_data_matrix, rowvar=False) for centered_data_matrix in centered_data_matrices]

eigenvalue_lists = []
eigenvector_lists = []

for covariance_matrix in covariance_matrices:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalue_lists.append(eigenvalues)
    eigenvector_lists.append(eigenvectors)




sorted_indices_lists = [np.argsort(eigenvalues)[::-1] for eigenvalues in eigenvalue_lists]
for i in range(len(eigenvalue_lists)):
    eigenvalue_lists[i] = eigenvalue_lists[i][sorted_indices_lists[i]]
    eigenvector_lists[i] = eigenvector_lists[i][:, sorted_indices_lists[i]]



cumulative_energy_list = [np.cumsum(eigenvalues) / np.sum(eigenvalues) for eigenvalues in eigenvalue_lists]

num_components_list = [np.argmax(cumulative_energy >= 0.99) + 1 for cumulative_energy in cumulative_energy_list]


selected_eigenvectors_list = [eigenvectors[:, :num_components] for eigenvectors, num_components in zip(eigenvector_lists, num_components_list)]



test_image_path = r'E:\3rd semester\Linear Algebra\Python_Assignment\data\yalefaces\subject15\subject15.3.png'  
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (50, 50), interpolation=cv2.INTER_LINEAR)
test_flattened = test_image_resized.flatten()
centered_test_list = [test_flattened - mean_vector for mean_vector in all_mean_vectors]


projected_test_list = [np.dot(centered_test, selected_eigenvectors) for centered_test, selected_eigenvectors in zip(centered_test_list, selected_eigenvectors_list)]



# Back-project the projected test image to the original space
back_projected_test_list = [np.dot(projected_test, selected_eigenvectors.T) + mean_vector for projected_test, selected_eigenvectors,mean_vector in zip(projected_test_list,selected_eigenvectors_list,all_mean_vectors)]

for i in range(len(back_projected_test_list)):
    back_projected_test_list[i] = back_projected_test_list[i].reshape(test_image_resized.shape)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image_resized, cmap='gray')
plt.title("Original Test Image")


mse_list = [np.mean((test_image_resized - back_projected_test) ** 2) for  back_projected_test in back_projected_test_list]
min_mse_index = np.argmin(mse_list)
min_mse_value = mse_list[min_mse_index]
print("min loss",min_mse_value)
print(f'predicted to be subject{label_list[0][min_mse_index]}')
plt.subplot(1, 3, 3)
plt.imshow(back_projected_test_list[min_mse_index], cmap='gray')
plt.title(f'prediction:{label_list[0][min_mse_index]}')

plt.show()
