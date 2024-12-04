import os
import numpy as np
import cv2

src = 'D:\interesting_projects\mmsegmentation\datasets\labels'

dst = src+'_modified'
os.makedirs(dst,exist_ok=True)
for i in os.listdir(src):
    img_path = os.path.join(src,i)
    dst_img_path = os.path.join(dst,i)
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    # img[img == 0] = 1
    img[img == 255] = 0

    cv2.imwrite(dst_img_path,img)




# src = 'D:\interesting_projects\mmsegmentation\datasets\labels/'