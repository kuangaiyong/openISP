#!/usr/bin/python
import numpy as np

class CCM:
    'Color Correction Matrix'

    def __init__(self, img, ccm):
        self.img = img
        self.ccm = ccm

    def execute(self):
        # 向量化矩阵乘法: img (H,W,3) x ccm (3,3)^T + offset (3,)
        ccm_img = np.dot(self.img.astype(np.int32), self.ccm[:, 0:3].T) + self.ccm[:, 3]
        ccm_img = ccm_img / 1024
        self.img = np.clip(ccm_img, 0, 255).astype(np.uint8)
        return self.img
