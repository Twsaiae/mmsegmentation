# import os
#
# import cv2
# import numpy as np
# import math
#
# # 读取图像
#
# src = 'D:\interesting_projects\mmsegmentation\datasets\labels'
# a = 0
# b = 0
# img_num=0
# for i in os.listdir(src):
#     # image_path = 'D:\interesting_projects\mmsegmentation\datasets\labels/20240926155249519_4.bmp'  # 替换为你的标签图路径
#     image_path = os.path.join(src, i)
#     image = cv2.imread(image_path)
#     # image = cv2.imread(image_path,0)
#     # print(np.unique(image))
#
#     # 检查图像是否成功加载
#     if image is None:
#         raise ValueError("图像加载失败，请检查路径是否正确")
#
#     # 创建一个掩码，标记RGB值为(1, 1, 1)的像素
#     # mask = np.all(image == [0, 0, 0], axis=-1)
#     mask = np.all(image == [1, 1, 1], axis=-1)
#
#     # 计算掩码中True的数量（即RGB值为(1, 1, 1)的像素数量）
#     a_pixel_num = np.sum(mask)
#
#     # 计算图像的总像素数量
#     total_pixels = image.shape[0] * image.shape[1]
#
#
#     b_pixel_num = total_pixels-a_pixel_num
#
#
#     a+=a_pixel_num
#     b+=b_pixel_num
#     img_num+=1
#
# print([b/img_num,a/img_num])
#






# print(total/len)
# weight = math.log(total/len)
# weight2 = math.log(1-total/len)
#
#
# print(weight,weight2)

import numpy as np


def calculate_class_weights(pixel_counts, total_pixels, n_classes, epsilon=1.0):
    # 计算每个类别的像素占比
    p_i = np.array(pixel_counts) / total_pixels

    # 计算未经归一化的权重
    w_prime_i = 1.0 / np.log(p_i + epsilon)

    # 归一化权重，使得权重总和为n_classes
    w_i = w_prime_i / np.sum(w_prime_i) * n_classes

    return w_i


# 示例数据
pixel_counts = [2628693,11307]  # 每个类别的像素数量
total_pixels = sum(pixel_counts)  # 总像素数量
n_classes = len(pixel_counts)  # 类别数

# 计算权重
weights = calculate_class_weights(pixel_counts, total_pixels, n_classes)
print(weights)