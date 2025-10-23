"""
The primary function of the current code file is to perform targeted attacks on a specified convolutional neural network model using ASMP based on the L0 norm.
First, set `actual_images_folder_absolute_path` to the absolute path of the folder containing the original images.
Second, set `output_directory` to the absolute path of the folder where the adversarial examples will be saved.
Third, set `index_file_absolute_path` to the absolute path of the index file.
Fourth, set `weight_file_absolute_path` to the absolute path of the weight file.
Fifth, adjust the value of `tampering_ratio` to control the allowed ratio of tampered pixels.
Sixth, use `"from torchvision.models import googlenet"` to import the official convolutional neural network model structure, and adjust `model_current` to specify the current convolutional neural network model.
Finally, execute the code file to generate the targeted adversarial examples.
"""



import json
import shutil
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math
from skimage.metrics import structural_similarity as ssim
from skimage import io

def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3)

def calculate_ssim_for_folders(folder1, folder2):
    file_list1 = os.listdir(folder1)
    file_list2 = os.listdir(folder2)

    if len(file_list1) != len(file_list2):
        raise ValueError("The quantity in the two folders is not equal")

    ssim_results = []
    total_ssim = 0

    ssim_bins = {
        "0.995-1": 0,
        "0.996-1": 0,
        "0.997-1": 0,
        "0.998-1": 0,
        "0.999-1": 0
    }

    for file_name1, file_name2 in zip(file_list1, file_list2):
        if file_name1.endswith(('.png', '.jpg', '.jpeg', '.bmp')) and file_name2.endswith(
                ('.png', '.jpg', '.jpeg', '.bmp')):
            img1_path = os.path.join(folder1, file_name1)
            img2_path = os.path.join(folder2, file_name2)

            img1 = io.imread(img1_path)
            img2 = io.imread(img2_path)

            ssim_value = calculate_ssim(img1, img2)

            ssim_results.append((file_name1, file_name2, ssim_value))
            total_ssim += ssim_value

            if ssim_value >= 0.990:
                ssim_bins["0.995-1"] += 1
            if ssim_value >= 0.992:
                ssim_bins["0.996-1"] += 1
            if ssim_value >= 0.994:
                ssim_bins["0.997-1"] += 1
            if ssim_value >= 0.996:
                ssim_bins["0.998-1"] += 1
            if ssim_value >= 0.999:
                ssim_bins["0.999-1"] += 1

    print("\nASR:")
    for bin_name, count in ssim_bins.items():
        print(f"{bin_name}: {count*100/len(ssim_results)}%")

    return ssim_results

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eps_L0 = 0.2

# The maximum degree of tampering of a single pixel
degree = 255

# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

model_current = alexnet

# 指定目标类别（示例：ImageNet的'tabby cat'类别）
target_class = 281  # 修改为你的目标类别索引

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

model_CAM = alexnet(num_classes=1000)

success_num = 0
image_num = 0
for file_name in file_list:
    image_num = image_num + 1
    print(f"当前处理图像: {file_name}")
    actual_image_absolute_path = os.path.join(folder_path, file_name)
    image = Image.open(actual_image_absolute_path).resize((224, 224))

    ###################################################################
    model_CAM.load_state_dict(torch.load(weights_path, map_location=device))
    model_CAM.eval()
    img_cam = Image.open(actual_image_absolute_path).convert('RGB')

    img_tensor = data_transform(img_cam)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    actual_image = np.array(image)

    with torch.no_grad():
        output = model_CAM(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].numpy()

    target_category = [np.argmax(probabilities)]
    #####################################################
    target_layers = [model_CAM.features[-1]]
    ######################################################
    cam = GradCAM(model=model_CAM, target_layers=target_layers, use_cuda=False)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    flattened = grayscale_cam.flatten()
    threshold_index = int(len(flattened) * 0.1)
    threshold_value = np.partition(flattened, -threshold_index)[-threshold_index]
    mask_1 = (grayscale_cam >= threshold_value).astype(np.uint8)
    mask = np.zeros((3, 224, 224), dtype=np.uint8)
    mask[0, :, :] = mask_1
    mask[1, :, :] = mask_1
    mask[2, :, :] = mask_1
    ##########################################################

    # The R, G, B three channel matrix of the actual image
    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
    actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
    actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

    image = data_transform(image).unsqueeze(0).cuda()

    # Load image
    img_absolute_path = actual_image_absolute_path
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

    with open(json_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_current(num_classes=1000).to(device)

    # Load model weights
    weights_absolute_path = weights_path
    assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
    model.load_state_dict(torch.load(weights_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)

    # 原标签
    label = torch.tensor([top_indices[0]]).cuda()

    actual_image_index, x = predict_image_path(actual_image_absolute_path, json_absolute_path, weights_path, 0, model_current)

    # 有目标攻击：传入目标类别
    target_label = torch.tensor([target_class]).cuda()
    atk = SPLA(model, alpha=0.017, steps=max_num_iterative, mask=mask, targeted=True)  # 假设支持targeted模式
    adv_image = atk(image, target_label, actual_image_transform_matrix, actual_image_matrix, actual_image)

    image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("Targeted attack image.png")
    iterative_image_path = "Targeted attack image.png"

    attack_image_index, x = predict_image_path(iterative_image_path, json_absolute_path, weights_absolute_path,
                                               actual_image_index, model_current)
    if attack_image_index == target_class:
        success_num = success_num + 1
    success_rate = success_num / image_num
    print(f"success_rate: {success_rate * 100:.2f}%")
    actual_image_top1_index, x = predict_image_path(actual_image_absolute_path, json_absolute_path, weights_path, 0, model_current)
    print(f"Top-1 category index number of the actual image:{actual_image_top1_index}")

    adversarial_sample_image_top1_index, x = predict_image_path(iterative_image_path, json_absolute_path, weights_path,
                                                                0, model_current)
    print(f"Top-1 category index number of the adversarial sample:{adversarial_sample_image_top1_index} (target: {target_class})")

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
    image = Image.open(iterative_image_path)
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
    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[0], \
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[1], \
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[2] = pixel_weight_matrix_image_path(
        iterative_image_path, weights_path, adversarial_sample_image_top1_index,
        model_current
    )
    #######################################################################################
    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label = np.abs(
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label)
    # Flatten the matrix and calculate the absolute values
    flat_total_telationship = adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label.flatten()
    flat_abs_total_telationship = np.abs(flat_total_telationship)
    # Get the indices of the smallest 1000 absolute values
    smallest_indices = np.argpartition(flat_abs_total_telationship, 3 * 224 * 224 - 1)[:3 * 224 * 224]
    smallest_indices_sorted = smallest_indices[np.argsort(flat_abs_total_telationship[smallest_indices])]
    # Convert flat indices back to 3D indices (coordinates in the original matrix)
    index_matrix = np.array(
        [np.unravel_index(idx, adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label.shape) for idx
         in smallest_indices_sorted])

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
    # 记录上一轮优化后的对抗样本，防止本轮优化后不是对抗样本
    last_matrix = np.zeros((3, 224, 224), dtype=np.float64)

    while (right - left != 1):
        print(f"当前迭代次数：{iterative_num}")
        if flag == 1:
            last_matrix = change_matrix.copy()

        return_the_number_of_pixels = int(mid)
        change_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        change_matrix[0] = adversarial_sample_image_channel_R.copy()
        change_matrix[1] = adversarial_sample_image_channel_G.copy()
        change_matrix[2] = adversarial_sample_image_channel_B.copy()

        for i in range(return_the_number_of_pixels):
            change_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]] = \
                actual_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]]

        # Combine three channels into an RGB image
        image_rgb = np.stack([change_matrix[0], change_matrix[1], change_matrix[2]], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("targeted_untarget.png")  # 注意文件名
        adversarial_image_path = "targeted_untarget.png"

        iterative_image_current_top1_label, x = predict_image_path(adversarial_image_path, json_absolute_path,
                                                                   weights_path, 0, model_current)
        print("可能的优化标签",iterative_image_current_top1_label,x)
        if iterative_image_current_top1_label != target_class:  # 修改：检查是否达到目标类别
            flag = 0
            right = mid
            mid = (left + right) // 2
        else:
            flag = 1
            left = mid
            mid = (left + right) // 2

        iterative_num = iterative_num + 1

    # 保存最终优化后的对抗样本
    image_rgb = np.stack([last_matrix[0], last_matrix[1], last_matrix[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)
ssim_results = calculate_ssim_for_folders(actual_images_folder_absolute_path, output_directory)