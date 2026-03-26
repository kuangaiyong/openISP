#!/usr/bin/python
import numpy as np
from model.blc import get_bayer_slices


class WBGC:
    """Auto White Balance Gain Control

    Applies per-channel gain to compensate for illuminant color temperature.
    """

    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter      # [r_gain, gr_gain, gb_gain, b_gain]
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        r_gain, gr_gain, gb_gain, b_gain = self.parameter
        slices = get_bayer_slices(self.bayer_pattern)

        raw_h, raw_w = self.img.shape
        awb_img = np.empty((raw_h, raw_w), np.int16)

        gains = {'r': r_gain, 'gr': gr_gain, 'gb': gb_gain, 'b': b_gain}
        for channel, gain in gains.items():
            awb_img[slices[channel]] = self.img[slices[channel]] * gain

        self.img = awb_img
        return self.clipping()
