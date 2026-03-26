#!/usr/bin/python
import numpy as np

class GC:
    'Gamma Correction'

    def __init__(self, img, lut, mode):
        self.img = img
        self.lut = lut
        self.mode = mode

    def execute(self):
        if self.mode == 'rgb':
            # 将字典 LUT 转为 numpy 数组以支持向量化索引
            max_key = max(self.lut.keys())
            lut_array = np.zeros(max_key + 1, dtype=np.uint16)
            for k, v in self.lut.items():
                lut_array[k] = v

            # 将输入图像 clip 到 LUT 有效范围内
            img_clipped = np.clip(self.img, 0, max_key).astype(np.intp)
            gc_img = lut_array[img_clipped]
            gc_img = gc_img / 4  # 10bit -> 8bit

        elif self.mode == 'yuv':
            # YUV 模式下 lut 是包含两个 LUT 的元组/列表
            max_key_0 = max(self.lut[0].keys())
            max_key_1 = max(self.lut[1].keys())
            lut_array_0 = np.zeros(max_key_0 + 1, dtype=np.uint16)
            lut_array_1 = np.zeros(max_key_1 + 1, dtype=np.uint16)
            for k, v in self.lut[0].items():
                lut_array_0[k] = v
            for k, v in self.lut[1].items():
                lut_array_1[k] = v

            gc_img = np.empty_like(self.img, dtype=np.uint16)
            gc_img[:, :, 0] = lut_array_0[np.clip(self.img[:, :, 0], 0, max_key_0).astype(np.intp)]
            gc_img[:, :, 1] = lut_array_1[np.clip(self.img[:, :, 1], 0, max_key_1).astype(np.intp)]
            gc_img[:, :, 2] = lut_array_1[np.clip(self.img[:, :, 2], 0, max_key_1).astype(np.intp)]

        self.img = gc_img.astype(np.uint16)
        return self.img
