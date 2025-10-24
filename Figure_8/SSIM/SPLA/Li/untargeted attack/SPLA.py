import torch
import torch.nn as nn
from PIL import Image
import json
import torchvision.transforms as transforms
from torchvision.models import alexnet
from attack import Attack
import numpy as np
import z3
from smt_attack.constants import *
from scipy.linalg import eigh
import math

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

channel_clamps = {
    0: {"min": -2.1179, "max": 2.2489},  # 第 0 通道（R）
    1: {"min": -2.0357, "max": 2.4285},  # 第 1 通道（G）
    2: {"min": -1.8044, "max": 2.64}     # 第 2 通道（B）
}

def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()
    # 对概率值进行排序，返回排序后的结果和对应的原始索引
    sorted_values, sorted_indices = torch.sort(classification_probability, descending=True)

    # 获取第二大概率对应的原始下标（索引从0开始，所以取第1位）
    second_max_index = sorted_indices[1].item()
    print(index,classification_probability[index])
    print(second_max_index,classification_probability[second_max_index])
    return predicted_class_index, output[index].item()


def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):

    adv_image = actual_image_matrix.copy()


    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.detach().cpu().numpy()


    factors = np.array([[0.229, 0.485], [0.224, 0.456], [0.225, 0.406]])
    scales = np.array([255, 255, 255])


    for c in range(3):

        actual_transform = actual_image_transform_matrix[c]
        adversarial_transform = adversarial_sample_transform_matrix[c]
        actual_image = actual_image_matrix[c]


        mask_greater = adversarial_transform > actual_transform
        mask_less = adversarial_transform < actual_transform
        mask_equal = adversarial_transform == actual_transform


        adv_image[c] = np.where(mask_greater,
                                np.ceil((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                np.where(mask_less,
                                         np.floor((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                         np.where(mask_equal, actual_image, adv_image[c])))


    adv_image = np.clip(adv_image, 0, 255)

    return adv_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PCAIFGSM(Attack):
    # l2
    # def __init__(self, model, float(np.sqrt((8 ** 2) * 3 * 224 * 224)), alpha=1 / 255, steps=10, random_start=True, vcr=0.9, mask=np.zeros((3, 224, 224), dtype=np.float32)):
    def __init__(self, model, eps=255 / 255 * (2.4285 + 2.0357), alpha=1 / 255, steps=10, random_start=True, vcr=0.9, mask=np.zeros((3, 224, 224), dtype=np.float32)):
        super().__init__("IFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.vcr = vcr
        self.mask = mask
    def forward(self, images, labels,labels_top2, actual_image_transform_matrix, actual_image_matrix, actual_image):



        model_current = alexnet

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels_top2 = labels_top2.clone().detach().to(self.device)

        images.requires_grad = True



        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        mask = torch.from_numpy(self.mask).to(device)

        specific_values = [-0.017, 0.017]
        # specific_values = [-1, 1]
        adv_image_transform_matrix = actual_image_transform_matrix.copy()

        for _ in range(self.steps):

            print("当前迭代次数: ",_+1)
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost_top2 = loss(outputs, labels_top2)

            # 第一次计算梯度
            outputs = self.get_logits(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # 重新计算输出以生成新的计算图
            outputs = self.get_logits(adv_images)
            cost_top2 = loss(outputs, labels_top2)
            grad_top2 = torch.autograd.grad(
                cost_top2, adv_images, retain_graph=False, create_graph=False
            )[0]
            print("原始图像分类值",float(outputs[0][labels]),float(outputs[0][labels_top2]))
            grad = grad * mask
            grad_top2=grad_top2 * mask
            grad = grad.squeeze(0).cpu()
            grad_np = grad.cpu().numpy()
            grad_top2 = grad_top2.squeeze(0).cpu()
            grad_top2_np = grad_top2.cpu().numpy()
            solver = z3.Solver()
            variables = [[[0 for j in range(224)] for i in range(224)] for k in range(3)]

            while True:
                # 重置模型约束
                solver.reset()
                n = 0
                for k in range(3):
                    for i in range(224):
                        for j in range(224):
                            if self.mask[k][i][j] == 1:
                                # 创建变量
                                n += 1
                                var = z3.Real(f'x_{k}_{i}_{j}')
                                # var = z3.Int(f'x_{k}_{i}_{j}')
                                variables[k][i][j] = var
                                value_constraints = [variables[k][i][j] == val for val in specific_values]
                                solver.add(z3.Or(value_constraints))
                                solver.add(variables[k][i][j] * grad_np[k][i][j] > variables[k][i][j] * grad_top2_np[k][i][j])

                if solver.check() == z3.sat:
                    print(f"找到解")
                    model = solver.model()
                    # 1. 初始化对抗样本变换矩阵
                    adv_image_transform_matrix = adv_image_transform_matrix.copy()
                    for k in range(3):
                        for i in range(224):
                            for j in range(224):
                                var = variables[k][i][j]
                                if isinstance(var, z3.ExprRef):
                                    var_value = float(model[var].as_decimal(prec=10).rstrip('?'))
                                else:
                                    var_value = 0.0
                                adv_image_transform_matrix[k][i][j] = adv_image_transform_matrix[k][i][j] + var_value
                    m = (adv_image_transform_matrix == actual_image_transform_matrix)
                    print(np.sum(m))

                    for ch in range(3):
                        ch_min = channel_clamps[ch]["min"]
                        ch_max = channel_clamps[ch]["max"]
                        # NumPy 的 clip 函数对应 PyTorch 的 clamp，直接约束数值范围
                        adv_image_transform_matrix[ch] = np.clip(adv_image_transform_matrix[ch], ch_min, ch_max)

                    adv_images = torch.tensor(
                        adv_image_transform_matrix,
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    adv_images.requires_grad = True
                    break

            delta = adv_images - images

            # # Tampering intensity is 48
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.82198, max=0.82198)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.84032, max=0.84032)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.83659, max=0.83659)

            # Tampering intensity is 32
            delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.54799, max=0.54799)
            delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.56021, max=0.56021)
            delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.55772, max=0.55772)

            # # Tampering intensity is 20
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.34249, max=0.34249)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.35013, max=0.35013)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.34858, max=0.34858)

            # # Tampering intensity is 16
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.27399, max=0.27399)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.28010, max=0.28010)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.27886, max=0.27886)

            # # Tampering intensity is 12
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.20549, max=0.20549)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.21008, max=0.21008)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.20914, max=0.20914)

            # # Tampering intensity is 8
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.13699, max=0.13699)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.14005, max=0.14005)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.13941, max=0.13941)

            # # Tampering intensity is 6
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.10274, max=0.10274)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.10504, max=0.10504)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.10456, max=0.10456)

            # # Tampering intensity is 4
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.06849, max=0.06849)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.07002, max=0.07002)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.06970, max=0.06970)

            # # Tampering intensity is 2
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.03424, max=0.03424)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.03501, max=0.03501)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.03485, max=0.03485)

            # # Tampering intensity is 1
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.01712, max=0.01712)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.01750, max=0.01750)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.01742, max=0.01742)

            adv_image = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            mask_index = (self.mask == 0.0)

            adv_image[mask_index] = actual_image_matrix[mask_index]


            actual_delta = adv_image - actual_image_matrix
            # l2
            # actual_delta_L2 = np.linalg.norm(actual_delta)
            #
            # if actual_delta_L2 > self.eps:
            #     return previous_adv_image
            #
            # previous_adv_image = adv_image

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0], adv_image[1], adv_image[2],
                                                                            json_absolute_path, weights_path, 0,
                                                                            model_current)
            print("对抗样本标签",iterative_image_top1_label,x)
            if iterative_image_top1_label != labels:
                return adv_image

        return adv_image
