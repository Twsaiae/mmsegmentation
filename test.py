# from mmseg.apis import inference_segmentor, init_segmentor
import os
from tqdm import tqdm
from mmseg.apis import inference_model, init_model,show_result_pyplot
import mmcv

# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512/iter_2000.pth'



# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512/iter_20000.pth'


# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\segformer_mit-b0_8xb2-160k_ade20k-512x512/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_160000.pth'

# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\segformer_mit-b5_8xb2-160k_ade20k-512x512_test/segformer_mit-b5_8xb2-160k_ade20k-512x512_test.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\segformer_mit-b5_8xb2-160k_ade20k-512x512_test/iter_160000.pth'


# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024/iter_160000.pth'


# config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes.py'
# checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\pidnet-l_2xb6-120k_1024x1024-cityscapes/iter_120000.pth'

config_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py'
checkpoint_file = r'D:\interesting_projects\mmsegmentation\tools\work_dirs\ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/iter_120000.pth'

# 通过配置文件和模型权重文件构建模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 对单张图片进行推理并展示结果xl
# img = 'D:\interesting_projects\mmsegmentation\datasets\images/20240926162716653_38.bmp'  # or img = mmcv.imread(img), which will only load it once
src = r'C:\Users\Linn\Desktop\xian_data\test_all_src'  # or img = mmcv.imread(img), which will only load it once
# dst = src+'_dst_deeplabv3plus'
dst = src+'_dst_ddrnet_slim'
os.makedirs(dst,exist_ok=True)

for i in tqdm(os.listdir(src)):

    img = os.path.join(src,i)
    result = inference_model(model, img)
    # 在新窗口中可视化推理结果
    # model.show_result(img, result, show=True)

    # 或将可视化结果存储在文件中
    # 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
    # model.show_result(img, result, out_file='result.jpg', opacity=1)
    dst_img = os.path.join(dst,i)
    show_result_pyplot(model, img, result, out_file=dst_img, show=False)
    # show_result_pyplot(model, img, result, out_file=dst_img, show=True)
    # show_result_pyplot(model,img, result, out_file=dst_img, opacity=1)

