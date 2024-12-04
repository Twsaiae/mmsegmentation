_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',# 训练网络
    '../_base_/datasets/pascal_voc12.py',# 训练数据集
    '../_base_/default_runtime.py',# 训练的版本号
    # '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'# 训练逻辑和训练次数
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# 这里直接可以修改训练类别数目
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
