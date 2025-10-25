import json
import matplotlib.pyplot as plt
from torchvision.models import alexnet
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math


def sort_and_rank_matrix(matrix):
    channels, rows, cols = matrix.shape
    flattened_matrix = [(matrix[c, i, j], (c, i, j)) for c in range(channels) for i in range(rows) for j in range(cols)]
    flattened_matrix.sort(key=lambda x: x[0])
    result_matrix = np.zeros((channels, rows, cols), dtype=int)
    for rank, (_, (c, i, j)) in enumerate(flattened_matrix, start=1):
        result_matrix[c, i, j] = rank
    return result_matrix

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Input the image into the model for category prediction, and input it as the path of the image file
def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with open(index_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_cnn(num_classes=1000).to(device)

    # Load model weights
    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()

    return(predicted_class_index, output[index])

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Calculate the pixel weight matrix of a single specified image with category index number "index"
def pixel_weight_matrix_image_path(image_path, weight_path, index, model_cnn):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = model_cnn(num_classes=1000).to(device)

    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()

    return channel_r, channel_g, channel_b

def High_pass_filter(single_channel_matrix):
    single_channel_matrix_copy = single_channel_matrix.copy()
    f_transform = np.fft.fft2(single_channel_matrix_copy)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = single_channel_matrix_copy.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(np.sqrt(224 * 224 * 0.01 / pi))
    mask = np.ones((rows, cols), dtype=np.uint8)
    y, x = np.ogrid[:rows, :cols]
    center_mask = (x - ccol)**2 + (y - crow)**2 <= radius**2
    mask[center_mask] = 0
    filtered_f_shift = f_shift * mask
    filtered_f = np.fft.ifftshift(filtered_f_shift)
    filtered_image = np.fft.ifft2(filtered_f)
    filtered_image = np.abs(filtered_image)
    flattened_indices_desc = np.argsort(filtered_image.flatten())
    frequency_sorting_matrix = np.empty_like(filtered_image, dtype=int)
    for rank, index in enumerate(flattened_indices_desc, start=1):
        row, col = divmod(index, filtered_image.shape[1])
        frequency_sorting_matrix[row, col] = rank

    return frequency_sorting_matrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_current = alexnet
pi = math.pi

# The absolute path of index file
index_file_absolute_path = r"..."

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

# Import the folder path where the original input images are stored
actual_image_folders_absolute_path = r"..."

# Import the folder path where the unoptimized adversarial examples are stored.
adversarial_sample_image_folders_absolute_path = r"..."

# Import the path of the folder where the adversarial samples after streamlining, optimizing, and tampering with information are stored
optimized_adversarial_samples_folder_path = r"..."

file_list = os.listdir(actual_image_folders_absolute_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    print(file_name)
    actual_image_absolute_path = os.path.join(actual_image_folders_absolute_path, file_name)
    adversarial_sample_image_absolute_path = os.path.join(adversarial_sample_image_folders_absolute_path, file_name)

    # Perform category prediction on the adversarial_sample_image
    actual_image_top1_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path,
                                               weight_file_absolute_path, 0, model_current)
    print(f"Top-1 category index number of the actual image:{actual_image_top1_index}")


    # Perform category prediction on the original image
    adversarial_sample_image_top1_index, x = predict_image_path(adversarial_sample_image_absolute_path, index_file_absolute_path,
                                               weight_file_absolute_path, 0, model_current)
    print(f"Top-1 category index number of the adversarial sample:{adversarial_sample_image_top1_index}")


    # actual image
    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_channel_R = actual_image[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = actual_image[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = actual_image[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    # Standardize the actual image
    actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
    actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
    actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
    #############################################################################################
    # adversarial sample image
    image = Image.open(adversarial_sample_image_absolute_path)
    image = image.resize((224, 224))
    adversarial_sample_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    adversarial_sample_image_channel_R = adversarial_sample_image[:, :, 0]
    adversarial_sample_image_channel_R = adversarial_sample_image_channel_R.astype(np.float64)
    adversarial_sample_image_channel_G = adversarial_sample_image[:, :, 1]
    adversarial_sample_image_channel_G = adversarial_sample_image_channel_G.astype(np.float64)
    adversarial_sample_image_channel_B = adversarial_sample_image[:, :, 2]
    adversarial_sample_image_channel_B = adversarial_sample_image_channel_B.astype(np.float64)

    # Standardize the actual image
    adversarial_sample_image_transform_matrix_R = ((adversarial_sample_image_channel_R / 255) - 0.485) / 0.229
    adversarial_sample_image_transform_matrix_G = ((adversarial_sample_image_channel_G / 255) - 0.456) / 0.224
    adversarial_sample_image_transform_matrix_B = ((adversarial_sample_image_channel_B / 255) - 0.406) / 0.225

    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label = np.zeros((3, 224, 224), dtype=np.float64)
    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[0], adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[1], adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[2] = pixel_weight_matrix_image_path(
        adversarial_sample_image_absolute_path, weight_file_absolute_path, adversarial_sample_image_top1_index, model_current
    )
    #######################################################################################
    perturbation_information_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    perturbation_information_matrix[0] = adversarial_sample_image_transform_matrix_R - actual_image_transform_matrix_R
    perturbation_information_matrix[1] = adversarial_sample_image_transform_matrix_G - actual_image_transform_matrix_G
    perturbation_information_matrix[2] = adversarial_sample_image_transform_matrix_B - actual_image_transform_matrix_B
    #######################################################################################
    pixel_frequency_order = np.zeros((3, 224, 224), dtype=np.float64)
    pixel_frequency_order[0] = High_pass_filter(adversarial_sample_image_channel_R) / 224 / 224
    pixel_frequency_order[1] = High_pass_filter(adversarial_sample_image_channel_G) / 224 / 224
    pixel_frequency_order[2] = High_pass_filter(adversarial_sample_image_channel_B) / 224 / 224

    pixel_classification_contribution_value_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    pixel_classification_contribution_value_matrix[0] = perturbation_information_matrix[0] * adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[0]
    pixel_classification_contribution_value_matrix[1] = perturbation_information_matrix[1] * adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[1]
    pixel_classification_contribution_value_matrix[2] = perturbation_information_matrix[2] * adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[2]

    pixel_classification_contribution_value_matrix = sort_and_rank_matrix(np.abs(pixel_classification_contribution_value_matrix))
    hypotenuse = (pixel_classification_contribution_value_matrix**2 + pixel_frequency_order**2).copy()
    legs = np.abs(pixel_classification_contribution_value_matrix - pixel_frequency_order) / np.sqrt(2)
    total_telationship = hypotenuse - legs**2

    # Flatten the matrix and calculate the absolute values
    flat_total_telationship = total_telationship.flatten()
    flat_abs_total_telationship = np.abs(flat_total_telationship)
    # Get the indices of the smallest 1000 absolute values
    smallest_indices = np.argpartition(flat_abs_total_telationship, 3*224*224-1)[:3*224*224]
    smallest_indices_sorted = smallest_indices[np.argsort(flat_abs_total_telationship[smallest_indices])]
    # Convert flat indices back to 3D indices (coordinates in the original matrix)
    index_matrix = np.array([np.unravel_index(idx, total_telationship.shape) for idx in smallest_indices_sorted])

    change_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    change_matrix[0] = adversarial_sample_image_channel_R.copy()
    change_matrix[1] = adversarial_sample_image_channel_G.copy()
    change_matrix[2] = adversarial_sample_image_channel_B.copy()

    actual_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_matrix[0] = actual_image_channel_R.copy()
    actual_matrix[1] = actual_image_channel_G.copy()
    actual_matrix[2] = actual_image_channel_B.copy()

    iterative_num = 1
    flag = 1
    left = 0
    right = 3 * 224 * 224
    mid = (left + right) / 2
    last_matrix = np.zeros((3, 224, 224), dtype=np.float64)

    while(right - left != 1):
        print(f"Current number of iterations：{iterative_num}")
        if flag == 1:
            last_matrix = change_matrix.copy()

        return_the_number_of_pixels = int(mid)
        change_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        change_matrix[0] = adversarial_sample_image_channel_R.copy()
        change_matrix[1] = adversarial_sample_image_channel_G.copy()
        change_matrix[2] = adversarial_sample_image_channel_B.copy()

        for i in range(return_the_number_of_pixels):
            change_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]] = actual_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]]

        # Combine three channels into an RGB image
        image_rgb = np.stack([change_matrix[0], change_matrix[1], change_matrix[2]], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("n.png")
        iterative_image_path = "n.png"

        iterative_image_current_top1_label, x = predict_image_path(iterative_image_path, index_file_absolute_path,
                                                                   weight_file_absolute_path, 0, model_current)
        if iterative_image_current_top1_label == actual_image_top1_index:
            flag = 0
            right = mid
            mid = (left + right) // 2

        else:
            flag = 1
            left = mid
            mid = (left + right) // 2

        iterative_num = iterative_num + 1

    image_rgb = np.stack([last_matrix[0], last_matrix[1], last_matrix[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(optimized_adversarial_samples_folder_path, new_image_name)
    image_pil.save(new_image_path)