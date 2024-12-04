import mmseg
import mmcv
import torch
import cv2
print(torch.__version__)
print(mmcv.__version__)
print(mmseg.__version__)

# 这里全用了次顶配，就可以使用了
# 2.1.2+cu118
# 2.1.0
# 1.2.2




import cv2
import numpy as np


def click_event(event, x, y, flags, param):
    # 当左键点击时
    if event == cv2.EVENT_LBUTTONDOWN:
        # 显示点击的坐标
        print(x, ' ', y)

        # 显示点击位置的像素的RGB值
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        print("Red: ", red)
        print("Green: ", green)
        print("Blue: ", blue)


# 读取图片
# img = cv2.imread('your_image_path.jpg')
img = cv2.imread('D:\interesting_projects\mmsegmentation\datasets\labels/20240926155249519_4.bmp')
cv2.imshow('image', img)

# 设置鼠标回调函数
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()


# from mmseg.apis import MMSegInferencer
# # Load models into memory
# inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
# # Inference
# inferencer('demo/demo.png', show=True)



# python tools/train.py configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py
# python tools/train.py configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py