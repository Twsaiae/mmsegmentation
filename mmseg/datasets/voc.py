# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PascalVOCDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'tape_edge'),
        palette=[[0, 0, 0],[1, 1, 1]])
        # palette=[[255, 0, 0],[0, 0, 255]])

    def __init__(self,
                 ann_file,
                 # img_suffix='.jpg',
                 img_suffix='.bmp',
                 # seg_map_suffix='.png',
                 seg_map_suffix='.bmp',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        # assert fileio.exists(self.data_prefix['img_path'],
        #                      self.backend_args) and osp.isfile(self.ann_file)
        fileio.exists(self.data_prefix['img_path'],
                      self.backend_args) and osp.isfile(self.ann_file)
