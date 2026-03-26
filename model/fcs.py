#!/usr/bin/python
import numpy as np

class FCS:
    'False Color Suppresion'

    def __init__(self, img, edgemap, fcs_edge, gain, intercept, slope):
        self.img = img
        self.edgemap = edgemap
        self.fcs_edge = fcs_edge
        self.gain = gain
        self.intercept = intercept
        self.slope = slope

    def clipping(self):
        np.clip(self.img, 0, 255, out=self.img)
        return self.img

    def execute(self):
        abs_edge = np.abs(self.edgemap).astype(np.float64)

        # 向量化条件: 分三段计算 uvgain
        uvgain = np.where(
            abs_edge <= self.fcs_edge[0],
            self.gain,
            np.where(
                abs_edge < self.fcs_edge[1],
                self.intercept - self.slope * abs_edge,
                0
            )
        ).astype(np.float64)

        # 广播乘法: uvgain (H,W) * img (H,W,2) / 256 + 128
        fcs_img = uvgain[:, :, np.newaxis] * self.img.astype(np.float64) / 256 + 128
        self.img = fcs_img.astype(np.int16)
        return self.clipping()
